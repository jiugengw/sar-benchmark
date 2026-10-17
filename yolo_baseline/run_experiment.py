#!/usr/bin/env python3
"""Train a pretrained Ultralytics YOLO model and compare it with TerraMind."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import statistics
import sys
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sarbench.backbones import build_backbone
from sarbench.adapters import add_adapters
from sarbench.data import MEAN, STD
from sarbench.metrics import detection_metrics
from sarbench.twins import clean_image_ids
from sarbench.model import MultiTaskModel


CONFIDENCE = 0.25
MATCH_IOU = 0.5
MAP_CONFIDENCE = 0.001
IMAGE_SIZE = 512
F1_REFERENCE = 0.9331
MAP_REFERENCE = 0.2613


def combined_delta_m(macro_f1: float, map_score: float) -> float:
    """Benchmark mean relative change versus the fixed ViT + LoRA validation reference."""
    return 100.0 * 0.5 * (
        (macro_f1 - F1_REFERENCE) / F1_REFERENCE
        + (map_score - MAP_REFERENCE) / MAP_REFERENCE
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    elif path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            with child.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    else:
        raise FileNotFoundError(path)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def xywh_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detection_pr(gt: dict, predictions: list[dict], image_ids: list[int], names: list[str]) -> dict:
    """One-to-one class-aware P/R at the shared score and IoU thresholds."""
    gt_by_image_class: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    for ann in gt["annotations"]:
        gt_by_image_class[(int(ann["image_id"]), int(ann["category_id"]))].append(list(map(float, ann["bbox"])))
    pred_by_class: dict[int, list[dict]] = defaultdict(list)
    for pred in predictions:
        if float(pred["score"]) >= CONFIDENCE:
            pred_by_class[int(pred["category_id"])].append(pred)

    metrics = {}
    for category_id, name in enumerate(names, start=1):
        true_positives = false_positives = false_negatives = 0
        for image_id in image_ids:
            truths = gt_by_image_class.get((int(image_id), category_id), [])
            candidates = sorted(
                (pred for pred in pred_by_class[category_id] if int(pred["image_id"]) == int(image_id)),
                key=lambda item: float(item["score"]), reverse=True)
            matched: set[int] = set()
            for pred in candidates:
                overlaps = [(index, xywh_iou(list(map(float, pred["bbox"])), box))
                            for index, box in enumerate(truths) if index not in matched]
                best = max(overlaps, key=lambda item: item[1], default=(-1, 0.0))
                if best[1] >= MATCH_IOU:
                    matched.add(best[0])
                    true_positives += 1
                else:
                    false_positives += 1
            false_negatives += len(truths) - len(matched)
        metrics[name] = {
            "precision": true_positives / max(1, true_positives + false_positives),
            "recall": true_positives / max(1, true_positives + false_negatives),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
    return metrics


def image_class_metrics(rows_by_id: dict[int, dict], predictions: list[dict], image_ids: list[int], names: list[str]) -> dict:
    """Derive image class from the top >=0.25 detection; images without one are unknown/missed."""
    best: dict[int, dict] = {}
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        if float(prediction["score"]) < CONFIDENCE:
            continue
        if image_id not in best or float(prediction["score"]) > float(best[image_id]["score"]):
            best[image_id] = prediction

    confusion = [[0 for _ in range(len(names) + 1)] for _ in names]
    for image_id in image_ids:
        truth = int(rows_by_id[int(image_id)]["label"])
        prediction = best.get(int(image_id))
        predicted_class = int(prediction["category_id"]) - 1 if prediction else len(names)
        confusion[truth][predicted_class] += 1

    class_scores = {}
    for class_id, name in enumerate(names):
        tp = confusion[class_id][class_id]
        fn = sum(confusion[class_id]) - tp
        fp = sum(confusion[row][class_id] for row in range(len(names))) - tp
        class_scores[name] = {
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        }
    correct = sum(confusion[i][i] for i in range(len(names)))
    return {
        "method": "class of highest-confidence detection with score >= 0.25; no qualifying detection is an unknown and counts as incorrect",
        "confidence_threshold": CONFIDENCE,
        "accuracy": correct / max(1, len(image_ids)),
        "macro_f1": statistics.mean(value["f1"] for value in class_scores.values()),
        "balanced_accuracy": statistics.mean(value["recall"] for value in class_scores.values()),
        "no_detection_images": sum(confusion[row][len(names)] for row in range(len(names))),
        "per_class": class_scores,
        "confusion_matrix": confusion,
        "confusion_columns": names + ["unknown_no_detection"],
    }


def build_gt(source_gt: dict, image_ids: list[int]) -> dict:
    wanted = set(map(int, image_ids))
    return {
        **source_gt,
        "images": [image for image in source_gt["images"] if int(image["id"]) in wanted],
        "annotations": [ann for ann in source_gt["annotations"] if int(ann["image_id"]) in wanted],
    }


def to_original_box(xyxy: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = map(float, xyxy)
    x1, x2 = sorted((max(0.0, min(IMAGE_SIZE, x1)), max(0.0, min(IMAGE_SIZE, x2))))
    y1, y2 = sorted((max(0.0, min(IMAGE_SIZE, y1)), max(0.0, min(IMAGE_SIZE, y2))))
    sx, sy = width / IMAGE_SIZE, height / IMAGE_SIZE
    left, top, right, bottom = x1 * sx, y1 * sy, x2 * sx, y2 * sy
    return [left, top, right - left, bottom - top]


def load_baseline_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    backbone = build_backbone("terramind", pretrained=False)
    add_adapters(backbone, "moelora")
    model = MultiTaskModel(backbone, task_routing=True, anchor_profile="default")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def baseline_speed(paths: list[Path], checkpoint: Path, device: torch.device) -> dict:
    """Run full TerraMind + RPN/ROI, including grayscale decode and normalization, at batch size one."""
    model = load_baseline_model(checkpoint, device)

    def run_one(path: Path) -> None:
        with Image.open(path) as image:
            image = image.convert("L")
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).view(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)
        tensor = (tensor - MEAN) / STD
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, detections, _ = model(tensor)
            # Match YOLO's result handling: force detector outputs back to host memory.
            logits.argmax(dim=1).cpu().tolist()
            detections[0]["boxes"].cpu().tolist()
            detections[0]["labels"].cpu().tolist()
            detections[0]["scores"].cpu().tolist()

    for path in paths[:20]:
        run_one(path)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for path in paths:
        run_one(path)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    del model
    torch.cuda.empty_cache()
    return {
        "batch_size": 1,
        "images": len(paths),
        "wall_seconds": elapsed,
        "pipeline_ms_per_image": elapsed * 1000 / max(1, len(paths)),
        "pipeline_images_per_second": len(paths) / elapsed if elapsed else None,
        "scope": "grayscale PNG decode, project mean/std normalization, TerraMind + classification head + RPN/ROI postprocessing; batch size one; GPU warmup of 20 images; includes per-image host and GPU work",
    }


def draw_box(draw: ImageDraw.ImageDraw, xyxy: tuple[float, float, float, float], label: str, color: str, font) -> None:
    x1, y1, x2, y2 = xyxy
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    tx, ty = max(0, x1), max(0, y1 - 17)
    bounds = draw.textbbox((tx, ty), label, font=font)
    draw.rectangle(bounds, fill=color)
    draw.text((tx, ty), label, fill="black", font=font)


def write_visual_comparison(path: Path, dataset_root: Path, gt: dict, yolo_predictions: list[dict],
                            baseline_predictions: list[dict], image_map: dict, names: list[str],
                            model_name: str) -> list[int]:
    """Build class-stratified identical-image rows: ground truth, TerraMind, and YOLO."""
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in gt["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)
    yolo_by_image: dict[int, list[dict]] = defaultdict(list)
    tm_by_image: dict[int, list[dict]] = defaultdict(list)
    for pred in yolo_predictions:
        if float(pred["score"]) >= CONFIDENCE:
            yolo_by_image[int(pred["image_id"])].append(pred)
    for pred in baseline_predictions:
        if float(pred["score"]) >= CONFIDENCE:
            tm_by_image[int(pred["image_id"])].append(pred)
    info_by_id = {int(value["image_id"]): value for value in image_map.values()}

    sample_ids: list[int] = []
    used: set[int] = set()
    for category_id, _name in enumerate(names, start=1):
        candidates = [image_id for image_id, anns in anns_by_image.items()
                      if image_id not in used and any(int(ann["category_id"]) == category_id for ann in anns)]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda image_id: (len(anns_by_image[image_id]), image_id))
        sample_ids.append(chosen)
        used.add(chosen)

    cell = 370
    left = 95
    top = 50
    row_h = 410
    canvas = Image.new("RGB", (left + 3 * cell, top + max(1, len(sample_ids)) * row_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        row_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = title_font = row_font = ImageFont.load_default()
    for col, title in enumerate(("Ground truth", "TerraMind + RPN/ROI", model_name)):
        draw.text((left + col * cell + 8, 14), title, fill="black", font=title_font)

    name_by_id = {index + 1: name for index, name in enumerate(names)}
    for row_index, image_id in enumerate(sample_ids):
        info = info_by_id[image_id]
        relative = next(key for key, value in image_map.items() if int(value["image_id"]) == image_id)
        with Image.open(dataset_root / relative) as source:
            base = source.convert("RGB")
        primary_class = min((int(ann["category_id"]) for ann in anns_by_image[image_id]))
        draw.text((5, top + row_index * row_h + 15), f"{name_by_id[primary_class]}\nID {image_id}", fill="black", font=row_font)

        panels = []
        gt_panel = base.copy()
        d = ImageDraw.Draw(gt_panel)
        for ann in anns_by_image[image_id]:
            x, y, w, h = map(float, ann["bbox"])
            x *= IMAGE_SIZE / int(info["width"])
            w *= IMAGE_SIZE / int(info["width"])
            y *= IMAGE_SIZE / int(info["height"])
            h *= IMAGE_SIZE / int(info["height"])
            draw_box(d, (x, y, x + w, y + h), name_by_id[int(ann["category_id"])], "#36df74", font)
        panels.append(gt_panel)

        for model_predictions, output in ((tm_by_image[image_id], "tm"), (yolo_by_image[image_id], "yolo")):
            panel = base.copy()
            d = ImageDraw.Draw(panel)
            for prediction in model_predictions:
                x, y, w, h = map(float, prediction["bbox"])
                x *= IMAGE_SIZE / int(info["width"])
                w *= IMAGE_SIZE / int(info["width"])
                y *= IMAGE_SIZE / int(info["height"])
                h *= IMAGE_SIZE / int(info["height"])
                label = f"{name_by_id[int(prediction['category_id'])]} {float(prediction['score']):.2f}"
                draw_box(d, (x, y, x + w, y + h), label, "#ff554f", font)
            panels.append(panel)

        y0 = top + row_index * row_h + 34
        for col, panel in enumerate(panels):
            panel.thumbnail((cell - 12, cell - 12), Image.Resampling.LANCZOS)
            canvas.paste(panel, (left + col * cell + 5, y0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)
    return sample_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--results-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--classifier-checkpoint", type=Path,
                        help="supervised image-classification head trained from labels.csv")
    parser.add_argument("--model-label", help="display name for this architecture in the reports")
    parser.add_argument("--architecture", choices=("yolo", "rtdetr", "deformable_detr"), default="yolo",
                        help="Ultralytics detector family used for inference and feature extraction")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate an existing train-output/train/weights/best.pt checkpoint without retraining")
    args = parser.parse_args()

    from ultralytics import __version__ as ultralytics_version
    from detector_support import detection_head, feature_maps_from_hook, load_detector

    if not torch.cuda.is_available():
        raise RuntimeError("YOLO baseline must run on the allocated CUDA GPU")
    torch.set_num_threads(args.workers)
    device = torch.device("cuda:0")
    args.train_output.mkdir(parents=True, exist_ok=True)
    args.results_output.mkdir(parents=True, exist_ok=True)
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing local pretrained weights: {args.weights}")
    model_name = args.model_label or args.weights.stem.replace("yolo", "YOLO")

    split = json.loads(args.split.read_text())
    image_map = json.loads((args.dataset_root / "image_map.json").read_text())
    names = json.loads((args.dataset_root / "conversion_manifest.json").read_text())["classes"]
    image_ids = list(map(int, split["val_image_ids"]))
    info_by_id = {int(info["image_id"]): info for info in image_map.values()}
    if set(info_by_id) != set(map(int, split["train_image_ids"])) | set(image_ids):
        raise ValueError("Converted image map and saved TerraMind train/holdout split differ")
    val_paths = [args.dataset_root / f"images/val/{Path(info_by_id[image_id]['file_name']).stem}.png"
                 for image_id in image_ids]
    if not all(path.is_file() for path in val_paths):
        raise FileNotFoundError("A converted validation image is missing")

    train_dir = args.train_output
    # In eval-only mode, --weights remains the recorded pretrained source. The
    # trained checkpoint must come from the existing train-output directory.
    best_weights = train_dir / "train" / "weights" / "best.pt"
    training_seconds = None
    parameter_counts = None
    if not args.eval_only:
        model = load_detector(args.weights, args.architecture)
        parameter_counts = (
            sum(parameter.numel() for parameter in model.model.parameters()),
            sum(parameter.numel() for parameter in model.model.parameters() if parameter.requires_grad),
        )
        start = time.perf_counter()
        model.train(
            data=str(args.dataset_yaml.resolve()),
            epochs=args.epochs,
            imgsz=IMAGE_SIZE,
            batch=args.batch,
            workers=args.workers,
            device=0,
            pretrained=True,
            patience=args.epochs + 1,
            seed=0,
            deterministic=True,
            project=str(train_dir.resolve()),
            name="train",
            exist_ok=False,
            plots=True,
            val=True,
            cache=False,
        )
        training_seconds = time.perf_counter() - start
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if not best_weights.exists():
        raise FileNotFoundError(f"YOLO training completed without best.pt at {best_weights}")

    # Collect all holdout detections with a low score floor for full COCO AP.
    detector = load_detector(best_weights, args.architecture)
    classifier_head = None
    classifier_payload = None
    classifier_predictions: dict[int, int] = {}
    classifier_probabilities: dict[int, float] = {}
    classifier_feature_batches: list[torch.Tensor] = []
    classifier_hook = None
    classifier_embedded = False
    if args.classifier_checkpoint:
        from image_head import load_classifier_head, pool_detection_features

        if not args.classifier_checkpoint.is_file():
            raise FileNotFoundError(args.classifier_checkpoint)
        embedded_head = getattr(detector.model, "image_classifier", None)
        if embedded_head is not None:
            classifier_payload = torch.load(args.classifier_checkpoint, map_location="cpu", weights_only=True)
            embedded_head.load_state_dict(classifier_payload["state_dict"], strict=True)
            classifier_head = embedded_head.to(device).eval()
            classifier_embedded = True
        else:
            classifier_head, classifier_payload = load_classifier_head(args.classifier_checkpoint, device)
        classifier_head.eval()
        if classifier_payload.get("class_names") != names:
            raise ValueError("Image-classifier and detector class names differ")

        def capture_classifier_features(_module, inputs, output):
            features = feature_maps_from_hook(args.architecture, inputs, output)
            classifier_feature_batches.append(pool_detection_features(
                features, classifier_payload.get("feature_pooling", "mean")
            ).detach())

        classifier_hook = detection_head(detector.model).register_forward_hook(capture_classifier_features)
    if parameter_counts is None:
        # Ultralytics loads inference checkpoints with gradients disabled. The runner
        # trains every layer and does not pass a freeze option, so in eval-only mode
        # all model parameters were trainable during the original training run.
        total_parameters = sum(parameter.numel() for parameter in detector.model.parameters())
        trainable_parameters = total_parameters
    else:
        total_parameters, trainable_parameters = parameter_counts
    if classifier_head is not None and not classifier_embedded:
        classifier_parameters = sum(parameter.numel() for parameter in classifier_head.parameters())
        total_parameters += classifier_parameters
        trainable_parameters += classifier_parameters
    predict_args = dict(imgsz=IMAGE_SIZE, conf=MAP_CONFIDENCE, iou=0.7, max_det=300,
                        device=0, batch=1, workers=0, stream=True, verbose=False, save=False)
    # Warm up before measuring the shared-holdout pipeline speed.
    list(detector.predict(source=[str(val_paths[0])], **predict_args))
    classifier_feature_batches.clear()
    torch.cuda.synchronize(device)
    prediction_start = time.perf_counter()
    yolo_predictions: list[dict] = []
    speed_rows = []
    viz_by_id: dict[int, list[dict]] = defaultdict(list)
    image_id_by_stem = {
        Path(info["file_name"]).stem: image_id
        for image_id, info in info_by_id.items()
    }
    # A directory source respects batch=1. Passing a Python list makes Ultralytics
    # hold the whole list as a single in-memory batch on some loader paths.
    validation_source = args.dataset_root / "images" / "val"
    for result in detector.predict(source=str(validation_source), **predict_args):
        image_id = image_id_by_stem.get(Path(result.path).stem)
        if image_id is None:
            raise ValueError(f"Prediction returned an unknown validation image path: {result.path}")
        image_info = info_by_id[image_id]
        if classifier_head is not None:
            if len(classifier_feature_batches) != 1 or classifier_feature_batches[0].shape[0] != 1:
                raise RuntimeError("The image-classification head did not receive exactly one feature vector per image")
            with torch.inference_mode():
                logits = classifier_head(classifier_feature_batches.pop(0).to(device))
            probabilities = torch.softmax(logits[0], dim=0).detach().cpu().tolist()
            predicted_class = int(max(range(len(probabilities)), key=probabilities.__getitem__))
            classifier_predictions[image_id] = predicted_class
            classifier_probabilities[image_id] = float(probabilities[predicted_class])
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            scores = boxes.conf.detach().cpu().tolist()
            for coordinates, class_index, score in zip(xyxy, classes, scores):
                category_id = int(class_index) + 1
                bbox = to_original_box(coordinates, int(image_info["width"]), int(image_info["height"]))
                prediction = {"image_id": image_id, "category_id": category_id,
                              "bbox": bbox, "score": float(score)}
                yolo_predictions.append(prediction)
                x1, y1, x2, y2 = coordinates
                viz_by_id[image_id].append({"image_id": image_id, "category_id": category_id,
                                            "bbox": [float(x1) * image_info["width"] / IMAGE_SIZE,
                                                     float(y1) * image_info["height"] / IMAGE_SIZE,
                                                     float(x2 - x1) * image_info["width"] / IMAGE_SIZE,
                                                     float(y2 - y1) * image_info["height"] / IMAGE_SIZE],
                                            "score": float(score)})
        speed_rows.append(result.speed or {})
    if len(speed_rows) != len(image_ids) or {int(row["image_id"]) for row in yolo_predictions} - set(image_ids):
        raise RuntimeError(f"Holdout inference processed {len(speed_rows)} images; expected {len(image_ids)}")
    if classifier_hook is not None:
        classifier_hook.remove()
    torch.cuda.synchronize(device)
    yolo_elapsed = time.perf_counter() - prediction_start
    del detector, result
    gc.collect()
    torch.cuda.empty_cache()

    source_gt = json.loads((args.baseline_results / "instances_val_holdout.json").read_text())
    gt = build_gt(source_gt, image_ids)
    gt_path = args.results_output / "instances_val_holdout.json"
    gt_path.write_text(json.dumps(gt, indent=2) + "\n")
    (args.results_output / "predictions_val.json").write_text(json.dumps(yolo_predictions) + "\n")
    (args.results_output / "split.json").write_text(args.split.read_text())

    baseline_predictions = json.loads((args.baseline_results / "predictions_val.json").read_text())
    baseline_metrics_json = json.loads((args.baseline_results / "metrics.json").read_text())
    baseline_score = baseline_metrics_json.get("val", baseline_metrics_json.get("final_val", {}))
    yolo_det_metrics = detection_metrics(gt_path, yolo_predictions, image_ids)
    baseline_det_metrics = detection_metrics(gt_path, baseline_predictions, image_ids)
    yolo_pr = detection_pr(gt, yolo_predictions, image_ids, names)
    baseline_pr = detection_pr(gt, baseline_predictions, image_ids, names)

    with (args.data / "train" / "labels.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows_by_id = {int(row["image_id"]): row for row in rows}
    yolo_derived_cls = image_class_metrics(rows_by_id, yolo_predictions, image_ids, names)
    if classifier_head is not None:
        from image_head import classification_metrics

        yolo_cls = classification_metrics(
            [int(rows_by_id[image_id]["label"]) for image_id in image_ids],
            [classifier_predictions[image_id] for image_id in image_ids],
            names,
        )
        yolo_cls["method"] = (
            "jointly fine-tuned image-classification head on shared YOLO detection features"
            if classifier_embedded else
            "supervised image-classification head trained from labels.csv on frozen YOLO detection features"
        )
        yolo_cls["prediction_confidence"] = {
            "mean_top1_probability": statistics.mean(classifier_probabilities.values()),
            "minimum_top1_probability": min(classifier_probabilities.values()),
            "maximum_top1_probability": max(classifier_probabilities.values()),
        }
        (args.results_output / "image_classification_predictions.json").write_text(json.dumps({
            str(image_id): {"label": classifier_predictions[image_id],
                            "confidence": classifier_probabilities[image_id]}
            for image_id in image_ids
        }, indent=2) + "\n")
    else:
        yolo_cls = yolo_derived_cls

    # The same scores on the clean holdout subset: images with no near-duplicate twin among the training images.
    clean_ids = clean_image_ids(gt["images"])
    clean_det_metrics = detection_metrics(gt_path, yolo_predictions, clean_ids)
    if classifier_head is not None:
        clean_cls = classification_metrics(
            [int(rows_by_id[image_id]["label"]) for image_id in clean_ids],
            [classifier_predictions[image_id] for image_id in clean_ids],
            names,
        )
    else:
        clean_cls = image_class_metrics(rows_by_id, yolo_predictions, clean_ids, names)

    detection_class_policies = {"native_detector_classes": yolo_det_metrics}
    if classifier_head is not None:
        policy_candidates = [
            ("classifier_class_all_boxes", False),
            ("classifier_class_score_weighted", True),
        ]
        for policy, weight_score in policy_candidates:
            candidate_predictions = []
            for prediction in yolo_predictions:
                image_id = int(prediction["image_id"])
                class_index = classifier_predictions[image_id]
                confidence = classifier_probabilities[image_id]
                candidate = dict(prediction)
                candidate["category_id"] = class_index + 1
                if weight_score:
                    candidate["score"] = float(candidate["score"]) * confidence
                candidate_predictions.append(candidate)
            policy_metrics = detection_metrics(gt_path, candidate_predictions, image_ids)
            policy_metrics["delta_vs_native"] = {
                key: policy_metrics[key] - yolo_det_metrics[key]
                for key in ("mAP", "AP50", "AP75", "AP_small", "AR_small")
            } | {
                "per_class_AP": {
                    name: policy_metrics["per_class_AP"][name] - yolo_det_metrics["per_class_AP"][name]
                    for name in names
                },
            }
            detection_class_policies[policy] = policy_metrics
            (args.results_output / f"predictions_val_{policy}.json").write_text(
                json.dumps(candidate_predictions) + "\n"
            )
        detection_class_policies["comparison_note"] = (
            "All policies were evaluated on the same fixed holdout; no holdout-based submission policy was selected."
        )
    val_paths_by_id = [args.dataset_root / f"images/val/{Path(info_by_id[i]['file_name']).stem}.png" for i in image_ids]
    baseline_speed_path = args.baseline_results / "inference_speed_batch1.json"
    if baseline_speed_path.is_file():
        baseline_speed_result = json.loads(baseline_speed_path.read_text())
    else:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        minimum_free_bytes = 8 * 1024**3
        if free_bytes < minimum_free_bytes:
            baseline_speed_result = {
                "status": "deferred_due_to_gpu_memory",
                "available_gpu_memory_gib": free_bytes / 1024**3,
                "required_free_memory_gib": minimum_free_bytes / 1024**3,
                "reason": "A concurrent GPU workload left insufficient headroom for the TerraMind timing pass.",
            }
        else:
            try:
                baseline_speed_result = baseline_speed(val_paths_by_id, args.baseline_checkpoint, device)
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                baseline_speed_result = {
                    "status": "deferred_due_to_gpu_memory",
                    "available_gpu_memory_gib": free_bytes / 1024**3,
                    "required_free_memory_gib": minimum_free_bytes / 1024**3,
                    "reason": f"TerraMind timing exceeded available GPU memory: {error}",
                }
            else:
                baseline_speed_result["status"] = "completed"
                baseline_speed_path.write_text(json.dumps(baseline_speed_result, indent=2) + "\n")

    yolo_speed = {
        "batch_size": 1,
        "images": len(image_ids),
        "wall_seconds": yolo_elapsed,
        "pipeline_ms_per_image": yolo_elapsed * 1000 / max(1, len(image_ids)),
        "pipeline_images_per_second": len(image_ids) / yolo_elapsed if yolo_elapsed else None,
        "ultralytics_mean_stage_ms_per_image": {
            key: statistics.mean(float(row.get(key, 0.0)) for row in speed_rows) if speed_rows else None
            for key in ("preprocess", "inference", "postprocess")
        },
        "scope": "grayscale PNG decode, Ultralytics preprocessing, YOLO forward pass and NMS/postprocessing; batch size one; GPU warmup; pipeline wall time includes per-image host and GPU work",
    }

    # Class-stratified side-by-side artifact uses the same nine validation images for each model.
    visual_path = args.results_output / "visual_comparison.png"
    sample_ids = write_visual_comparison(visual_path, args.dataset_root, gt,
                                         viz_by_id_to_list(viz_by_id), baseline_predictions,
                                         image_map, names, model_name)

    try:
        import ultralytics
        ultralytics_git = getattr(ultralytics, "__version__", ultralytics_version)
    except ImportError:
        ultralytics_git = ultralytics_version
    result_val = {
        **yolo_det_metrics,
        "precision_recall_at_conf_0_25_iou_0_50": yolo_pr,
        "accuracy": yolo_cls["accuracy"],
        "macro_f1": yolo_cls["macro_f1"],
        "combined_delta_m": combined_delta_m(yolo_cls["macro_f1"], yolo_det_metrics["mAP"]),
        "balanced_accuracy": yolo_cls["balanced_accuracy"],
        "image_classification": yolo_cls,
        "speed": yolo_speed,
        "clean_holdout": {
            "images": len(clean_ids),
            **clean_det_metrics,
            "accuracy": clean_cls["accuracy"],
            "macro_f1": clean_cls["macro_f1"],
            "combined_delta_m": combined_delta_m(clean_cls["macro_f1"], clean_det_metrics["mAP"]),
            "balanced_accuracy": clean_cls["balanced_accuracy"],
            "image_classification": clean_cls,
        },
    }
    if classifier_head is not None:
        result_val["detection_derived_image_classification"] = yolo_derived_cls
        result_val["detection_class_policies"] = detection_class_policies
    settings_file = train_dir / "train" / "args.yaml"
    if not settings_file.is_file():
        settings_file = next(
            (candidate for candidate in (
                train_dir / "experiment_config.json",
                train_dir / "benchmark_settings.json",
            ) if candidate.is_file()),
            settings_file,
        )
    report_metadata = {
        "python": platform.python_version(),
        "libraries": {"torch": torch.__version__, "ultralytics": ultralytics_git},
        "git_commit": git_revision(),
        "configuration": {
            "backbone": model_name,
            "detector_architecture": args.architecture,
            "adapt": "native pretrained backbone, neck, and detection head",
            "training_scheme": (
                "joint detection and image-classification fine-tuning"
                if classifier_embedded else "full fine-tuning; no LoRA and no frozen layers"
            ),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "epochs": args.epochs,
            "image_size": IMAGE_SIZE,
            "batch_size": args.batch,
            "workers": args.workers,
            "seed": 0,
            "pretrained_weights": str(args.weights),
            "dataset_yaml": str(args.dataset_yaml),
            "split": str(args.split),
            "ultralytics_version": ultralytics_git,
            "validation_image_ids": len(image_ids),
            "precision_recall_confidence": CONFIDENCE,
            "precision_recall_iou": MATCH_IOU,
            "image_classification_head": str(args.classifier_checkpoint) if args.classifier_checkpoint else None,
            "image_classification_training": (
                "jointly fine-tuned from labels.csv; shared detector features"
                if classifier_embedded else "supervised from labels.csv; frozen detector features"
            ),
            "evaluated_box_class_policies": [
                key for key, value in detection_class_policies.items()
                if isinstance(value, dict) and "mAP" in value
            ],
        },
    }
    baseline_comparison = {
        "run": "TerraMind + MoE-LoRA + RPN/ROI, job 326894",
        "detection": baseline_det_metrics,
        "precision_recall_at_conf_0_25_iou_0_50": baseline_pr,
        "image_classification": {
            "accuracy": baseline_score.get("accuracy"),
            "macro_f1": baseline_score.get("macro_f1"),
            "balanced_accuracy": baseline_score.get("balanced_accuracy"),
            "method": "TerraMind image classification head (native classification predictions)",
        },
        "speed": baseline_speed_result,
    }
    summary = {
        "report_metadata": report_metadata,
        "args": report_metadata["configuration"],
        "val": result_val,
        "baseline_comparison": baseline_comparison,
        "training": {
            "completed_epochs": args.epochs,
            "training_seconds": training_seconds,
            "training_seconds_note": "null for evaluation-only recovery jobs; the original training job completed its configured epochs",
            "best_weights": str(best_weights),
            "pretrained_weights_sha256": sha256(args.weights),
            "best_weights_sha256": sha256(best_weights),
            "classifier_head_checkpoint": str(args.classifier_checkpoint) if args.classifier_checkpoint else None,
            "ultralytics_version": ultralytics_git,
            "torch_version": torch.__version__,
            "cuda_name": torch.cuda.get_device_name(device),
            "settings_file": str(settings_file),
        },
        "split": {"train_images": len(split["train_image_ids"]), "validation_images": len(image_ids),
                  "source": str(args.split)},
        "visual_comparison": {"path": str(visual_path), "same_validation_image_ids": sample_ids},
        "predictions_file": str(args.results_output / "predictions_val.json"),
        "ground_truth_file": str(gt_path),
        "timing_note": "Speed measurements use batch size one and the same converted holdout PNGs. Each pipeline includes image decode, preprocessing, inference, and postprocessing. Twenty images are used for warmup. TerraMind timing is explicitly marked deferred if concurrent GPU memory prevents a valid measurement.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.results_output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.results_output / "experiment.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.train_output / "experiment.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps({
        model_name: {"mAP": yolo_det_metrics["mAP"], "AP50": yolo_det_metrics["AP50"],
                    "image_accuracy": yolo_cls["accuracy"], "image_macro_f1": yolo_cls["macro_f1"],
                    "pipeline_ms_per_image": yolo_speed["pipeline_ms_per_image"]},
        f"{model_name}_clean_holdout": {key: result_val["clean_holdout"][key]
                                        for key in ("images", "mAP", "AP50", "macro_f1", "combined_delta_m")},
        "TerraMind_baseline": {"mAP": baseline_det_metrics["mAP"], "AP50": baseline_det_metrics["AP50"],
                               "accuracy": baseline_score.get("accuracy"),
                               "pipeline_ms_per_image": baseline_speed_result["pipeline_ms_per_image"]},
        "result_dir": str(args.results_output),
        "visual_comparison": str(visual_path),
    }, indent=2))


def viz_by_id_to_list(viz_by_id: dict[int, list[dict]]) -> list[dict]:
    return [item for image_predictions in viz_by_id.values() for item in image_predictions]


if __name__ == "__main__":
    main()
