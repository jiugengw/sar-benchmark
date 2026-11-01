#!/usr/bin/env python3
"""Create a project-format prediction ZIP (classification.csv + detection.json) for the course images.

By default: one detector pass, with an optional single image-classifier head. Every ZIP we made starts from it.

With --final: our final submission (Δm +65.80, the flip soup without refits) from the saved weights in final_model/,
training nothing, in five steps:
  1. detector boxes: the default pass, one2many branch, boxes >= 0.001, at most 300 per image
  2. rounding to whole pixels: round_boxes.py (whole-pixel-classes)
  3. hedging: resized and shifted copies of each box at lower scores (hedge_study.py)
  4. low-score tail: the detector's boxes below 0.001, ranked last (the default pass with --conf 1e-5 --max-det 1000,
     then lowconf_tail.py, which checks that the boxes >= 0.001 are the ones of step 1)
  5. image labels: the saved two-head classifier (branch head + conv head, probabilities averaged 50/50, highest class).
     The size rule of classifier_combo_study.py is left out (30 Oct): it changed 0 of 939 holdout and 0 of 1,426 test
     labels, and on images from new sources it could force a wrong label.
Each GPU step runs in its own process, as when the uploaded ZIP was made, so the result matches that ZIP.

Run from the repository root on a CUDA GPU, after the setup in docs/TEAM_GUIDE.md section 1, with the weights in
final_model/ (final_model/README.md says where to download them):
    python models/yolo/make_submission.py --final --out final.zip                     # Phase 1 images (data/course-data/val)
    python models/yolo/make_submission.py --final --images NEW_DIR --out phase2.zip   # NEW_DIR: images.json + the images
    python models/yolo/make_submission.py --final --out final.zip --compare submissions/alldata-s1fs-soup6-rounded-hedged-s1two-tail/alldata-s1fs-soup6-rounded-hedged-s1two-tail.zip
NEW_DIR is laid out like data/course-data/val: an images.json whose "images" list id, file_name, width and height,
and the image files at NEW_DIR/<file_name>. Intermediate ZIPs and probabilities are kept in <out>-work/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from input_filters import active_filter, apply as apply_input_filter


IMAGE_SIZE = 512
CLASSIFICATION_CONFIDENCE = 0.25
FINAL_WEIGHTS = Path("final_model")


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


def model_names(names) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index]) for index in range(len(names))]
    return [str(name) for name in names]


def force_multi_label() -> None:
    """Make Ultralytics' NMS keep every class above the score threshold for each box (multi_label=True, what its own
    validator uses) instead of only the box's top class."""
    import functools
    import importlib

    for name in ("ultralytics.utils.nms", "ultralytics.utils.ops"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, "non_max_suppression"):
            module.non_max_suppression = functools.partial(module.non_max_suppression, multi_label=True)


def run(*args) -> None:
    subprocess.run([sys.executable, "-u", *map(str, args)], check=True)


def read_zip(path: Path) -> tuple[list, dict]:
    with zipfile.ZipFile(path) as archive:
        boxes = json.loads(archive.read("detection.json"))
        rows = archive.read("classification.csv").decode().strip().splitlines()[1:]
    return boxes, dict(row.split(",") for row in rows)


def pair_boxes(boxes: list, ref_boxes: list, tolerance: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Pair boxes of the same image and class whose edges are all within `tolerance` px, closest pairs first.
    Returns the largest edge difference and the score difference of each pair."""
    groups = defaultdict(lambda: ([], []))
    for side, items in ((0, boxes), (1, ref_boxes)):
        for p in items:
            groups[p["image_id"], p["category_id"]][side].append(p)
    edges, scores = [], []
    for mine, ref in groups.values():
        if not mine or not ref:
            continue
        gap = np.abs(np.array([p["bbox"] for p in mine])[:, None] - np.array([p["bbox"] for p in ref])[None]).max(2)
        used_mine, used_ref = set(), set()
        for i, j in zip(*np.unravel_index(np.argsort(gap, axis=None), gap.shape)):
            if gap[i, j] > tolerance + 1e-6:
                break
            if i in used_mine or j in used_ref:
                continue
            used_mine.add(i)
            used_ref.add(j)
            edges.append(gap[i, j])
            scores.append(abs(mine[i]["score"] - ref[j]["score"]))
    return np.array(edges), np.array(scores)


def compare(mine: Path, reference: Path) -> None:
    """Print how far the new ZIP is from a reference ZIP: MATCH (identical), CLOSE MATCH (same labels, boxes differ
    only by a different GPU's rounding) or DIFFERENT."""
    boxes, labels = read_zip(mine)
    ref_boxes, ref_labels = read_zip(reference)
    changed = sum(labels.get(i) != k for i, k in ref_labels.items())
    print(f"labels: {changed} of {len(ref_labels)} images differ from the reference")
    if boxes == ref_boxes:
        print(f"boxes: identical to the reference ({len(boxes)} boxes, same order and scores)")
        print("MATCH" if not changed else "DIFFERENT", flush=True)
        return
    # Another GPU type (or driver) gives the same model slightly different floating-point results: box edges move by
    # about 0.001 px, whole-pixel rounding can then flip by 1 px, and boxes at the score cut-offs can come or go.
    edges, scores = pair_boxes(boxes, ref_boxes)
    unpaired = max(len(boxes), len(ref_boxes)) - len(edges)
    print(f"boxes: {len(boxes)} vs {len(ref_boxes)} in the reference; {len(edges)} pair up within 1 px "
          f"(median edge difference {np.median(edges):.1e} px, median score difference {np.median(scores):.1e}, "
          f"largest {scores.max():.1e}); {unpaired} without a partner")
    close = not changed and unpaired <= 0.01 * len(ref_boxes) and np.median(edges) < 0.01
    print("CLOSE MATCH: same labels; boxes differ only by GPU rounding" if close else "DIFFERENT", flush=True)


def make_final(args: argparse.Namespace) -> None:
    """--final: the five steps of our final submission (see the top of this file)."""
    root = Path.cwd()
    if not (root / "models/yolo/round_boxes.py").is_file():
        sys.exit("Run from the repository root.")
    sys.path[:0] = [str(root / "models/yolo"), str(root)]
    from hedge_study import rewrite_zip as hedge_zip
    from lowconf_tail import main as add_tail
    from round_boxes import rewrite_zip as round_zip

    detector = args.weights or FINAL_WEIGHTS / "detector.pt"
    images_dir = args.images.resolve()
    images_json = images_dir / "images.json"
    images = json.loads(images_json.read_text())["images"]
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    work = args.out.with_name(args.out.stem + "-work")
    work.mkdir(parents=True, exist_ok=True)
    for name, path in (("detector", detector), ("branch", args.branch), ("conv", args.conv)):
        if not path.is_file():
            raise FileNotFoundError(f"{path} (download the weights: final_model/README.md)")
        print(f"{name}: {path} sha256 {sha256(path)}", flush=True)
    print(f"{len(images)} images from {images_dir}", flush=True)
    # The model was trained and packaged without an input filter; make sure none leaks in from the environment.
    os.environ["SAR_INPUT_FILTER"] = "none"
    os.environ["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "models/yolo"), os.environ.get("PYTHONPATH", "")])

    # The default pass reads <data>/val (the images) and <data>/train/labels.csv (a fallback label it needs to start)
    data = work / "data"
    data.mkdir(exist_ok=True)
    for name, target in (("val", images_dir), ("train", root / "data/course-data/train")):
        link = data / name
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target)

    base, rounded, hedged = work / "base.zip", work / "rounded.zip", work / "hedged.zip"
    lowconf, tailed = work / "lowconf.zip", work / "tailed.zip"
    for path in (base, rounded, hedged, lowconf, tailed):
        path.unlink(missing_ok=True)
    default_pass = Path(__file__).resolve()
    multi_label = ["--multi-label"] if args.multi_label else []
    print("== 1. detector boxes", flush=True)
    run(default_pass, "--weights", detector, "--data", data, "--detection-branch", "one2many", "--out", base,
        *multi_label)
    print("== 2. rounding", flush=True)
    round_zip(base, rounded, "whole-pixel-classes", images_json)
    print("== 3. hedging", flush=True)
    hedge_zip(rounded, hedged)
    print("== 4. low-score tail", flush=True)
    run(default_pass, "--weights", detector, "--data", data, "--detection-branch", "one2many", "--out", lowconf,
        "--conf", "1e-5", "--max-det", "1000", *multi_label)
    add_tail(hedged, lowconf, tailed, images_json)
    print("== 5. two-head classifier", flush=True)
    probs = {}
    for name, script, weights in (("branch", "branch_head_predict.py", args.branch),
                                  ("conv", "conv_head_predict.py", args.conv)):
        probs[name] = work / f"probs_{name}.json"
        run(root / "models/yolo" / script, "--detector", detector, "--branch", weights,
            "--images", images_json, "--split-root", images_dir, "--out", probs[name])
    b, c = (json.loads(probs[name].read_text()) for name in ("branch", "conv"))
    ids = [im["id"] for im in images]
    mean = 0.5 * np.array([b[str(i)] for i in ids]) + 0.5 * np.array([c[str(i)] for i in ids])
    labels = mean.argmax(1)

    with zipfile.ZipFile(tailed) as archive:
        detection = archive.read("detection.json")
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("detection.json", detection)
        archive.writestr("classification.csv", "image_id,label\n" + "".join(f"{i},{int(k)}\n" for i, k in zip(ids, labels)))
    print(f"FINAL_ZIP {args.out}: {len(json.loads(detection))} boxes, {len(ids)} image labels", flush=True)
    if args.compare:
        compare(args.out, args.compare)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", "--detector", type=Path,
                        help="trained Ultralytics checkpoint (with --final: default final_model/detector.pt)")
    parser.add_argument("--data", type=Path, default=Path("data/course-data"))
    parser.add_argument("--classes", type=Path, default=Path("data/yolo11s-sar-512/conversion_manifest.json"))
    parser.add_argument("--out", type=Path, required=True, help="output ZIP; existing files are not overwritten")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--architecture", choices=("yolo", "rtdetr", "deformable_detr"), default="yolo",
                        help="Ultralytics detector family used by the checkpoint")
    parser.add_argument("--classifier-checkpoint", type=Path,
                        help="optional supervised image-classification head trained on detector features")
    parser.add_argument("--detection-branch", choices=("default", "one2one", "one2many"), default="default")
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--conf", type=float, default=0.001, help="lowest box score kept (studies only; ZIPs use 0.001)")
    parser.add_argument("--max-det", type=int, default=300, help="most boxes kept per image")
    parser.add_argument("--multi-label", action="store_true",
                        help="keep every class above --conf for each box, as Ultralytics' validator does, not only the "
                             "top one (holdout, 1 Nov: no gain, so our ZIPs leave it off)")
    parser.add_argument("--box-class-policy", choices=(
        "native_detector_classes", "classifier_class_all_boxes", "classifier_class_score_weighted"
    ), default="native_detector_classes")
    final = parser.add_argument_group("final submission")
    final.add_argument("--final", action="store_true", help="make our final submission ZIP: the five steps above")
    final.add_argument("--images", type=Path, default=Path("data/course-data/val"),
                       help="folder with images.json and the images (default: the Phase 1 test images)")
    final.add_argument("--branch", type=Path, default=FINAL_WEIGHTS / "branch_head.pt", help="branch head weights")
    final.add_argument("--conv", type=Path, default=FINAL_WEIGHTS / "conv_head.pt", help="conv head weights")
    final.add_argument("--compare", type=Path, help="reference ZIP to compare the result with")
    args = parser.parse_args()
    if args.final:
        make_final(args)
        return
    if args.weights is None:
        parser.error("--weights is required (or use --final)")
    branch_nms = None if args.detection_branch == "default" else args.detection_branch == "one2many"

    import torch
    from detector_support import detection_head, feature_maps_from_hook, load_detector

    if not torch.cuda.is_available():
        raise RuntimeError("YOLO submission prediction requires the allocated CUDA GPU")
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    if args.classifier_checkpoint and not args.classifier_checkpoint.is_file():
        raise FileNotFoundError(args.classifier_checkpoint)
    if args.box_class_policy != "native_detector_classes" and not args.classifier_checkpoint:
        raise ValueError("A classifier checkpoint is required for classifier-based box class policies")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")

    class_names = json.loads(args.classes.read_text())["classes"]
    split_root = args.data / "val"
    images = json.loads((split_root / "images.json").read_text())["images"]
    if not images:
        raise ValueError("The official validation split is empty")
    if len({int(image["id"]) for image in images}) != len(images):
        raise ValueError("Official validation image IDs are not unique")
    if len({image["file_name"] for image in images}) != len(images):
        raise ValueError("Official validation image paths are not unique")

    with (args.data / "train" / "labels.csv").open(newline="") as stream:
        train_labels = [int(row["label"]) for row in csv.DictReader(stream)]
    prior_counts = Counter(train_labels)
    if not prior_counts:
        raise ValueError("Cannot derive an image-class fallback from training labels")
    fallback_class = prior_counts.most_common(1)[0][0]
    if not 0 <= fallback_class < len(class_names):
        raise ValueError(f"Training fallback class {fallback_class} is outside the class list")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip = args.out.with_suffix(args.out.suffix + ".tmp")
    if temporary_zip.exists():
        raise FileExistsError(f"Refusing to overwrite temporary archive {temporary_zip}")

    detector = load_detector(args.weights, args.architecture)
    if args.multi_label:
        force_multi_label()
    if args.detection_branch != "default":
        if args.architecture != "yolo":
            raise ValueError("Explicit branch selection is supported only for YOLO")
        head = detection_head(detector.model)
        if args.detection_branch == "one2many" and getattr(head, "cv2", None) is None:
            raise ValueError("This checkpoint has already removed its one-to-many branch")
        if args.detection_branch == "one2one" and getattr(head, "one2one_cv2", None) is None:
            raise ValueError("This checkpoint has no one-to-one branch")
        detector.model.end2end = args.detection_branch == "one2one"
    if model_names(detector.names) != class_names:
        raise ValueError(f"Checkpoint classes {model_names(detector.names)} do not match {class_names}")
    device = torch.device("cuda:0")
    classifier_head = None
    classifier_feature_batches: list[torch.Tensor] = []
    classifier_hook = None
    if args.classifier_checkpoint:
        from image_head import load_classifier_head, pool_detection_features

        classifier_head, classifier_payload = load_classifier_head(args.classifier_checkpoint, device)
        if classifier_payload.get("class_names") != class_names:
            raise ValueError("Image-classifier and detector class names differ")

        def capture_classifier_features(_module, inputs, output):
            feature_maps = feature_maps_from_hook(args.architecture, inputs, output)
            classifier_feature_batches.append(pool_detection_features(
                feature_maps, classifier_payload.get("feature_pooling", "mean")
            ).detach().cpu())

        # Register before Ultralytics creates its predictor on the first call;
        # predictor setup can replace the model object and lose later hooks.
        classifier_hook = detection_head(detector.model).register_forward_hook(
            capture_classifier_features
        )

    detections: list[dict] = []
    labels_by_id: dict[int, int] = {}
    probabilities_by_id: dict[int, list[float]] = {}  # the image classifier's class probabilities, saved beside the ZIP
    image_by_temp_stem: dict[str, dict] = {}
    active_classifier_probabilities: list[list[float]] = []
    active_classifier_index = 0
    start = time.perf_counter()
    input_filter = active_filter()
    print(f"Input filter: {input_filter}", flush=True)
    with tempfile.TemporaryDirectory(prefix="yolo-course-val-512-") as temp_dir:
        temp_root = Path(temp_dir)
        for image in images:
            image_id = int(image["id"])
            source = split_root / image["file_name"]
            if not source.is_file():
                raise FileNotFoundError(source)
            stem = f"{image_id:08d}"
            image_by_temp_stem[stem] = image
            with Image.open(source) as original:
                resized = original.convert("L").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
                )
                if input_filter != "none":  # the same fixed filter the detector was trained with
                    resized = Image.fromarray(apply_input_filter(input_filter, np.asarray(resized)), "RGB")
                resized.save(temp_root / f"{stem}.png", format="PNG")

        if classifier_head is not None:
            # Warm up Ultralytics before installing the feature hook so its dummy
            # batch cannot be mistaken for a real image's classifier features.
            list(detector.predict(
                source=Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)), imgsz=IMAGE_SIZE, batch=1, workers=0, device=0,
                conf=0.001, iou=args.nms_iou, max_det=300, stream=True, verbose=False, save=False,
                nms=branch_nms,
            ))
            classifier_feature_batches.clear()

        predictions = detector.predict(
            source=str(temp_root),
            imgsz=IMAGE_SIZE,
            batch=args.batch,
            workers=args.workers,
            device=0,
            conf=args.conf,
            iou=args.nms_iou,
            nms=branch_nms,
            max_det=args.max_det,
            stream=True,
            verbose=False,
            save=False,
        )
        processed = 0
        for result in predictions:
            stem = Path(result.path).stem
            image = image_by_temp_stem.get(stem)
            if image is None:
                raise ValueError(f"Unexpected prediction path from Ultralytics: {result.path}")
            image_id = int(image["id"])
            processed += 1
            classifier_class = None
            classifier_confidence = None
            if classifier_head is not None:
                if active_classifier_index >= len(active_classifier_probabilities):
                    if not classifier_feature_batches:
                        raise RuntimeError("The image-classification head did not receive a feature batch")
                    features = classifier_feature_batches.pop(0).to(device)
                    with torch.inference_mode():
                        probabilities = torch.softmax(classifier_head(features), dim=1).cpu().tolist()
                    if not probabilities:
                        raise RuntimeError("The image-classification head returned an empty feature batch")
                    active_classifier_probabilities = probabilities
                    active_classifier_index = 0
                if active_classifier_index >= len(active_classifier_probabilities):
                    raise RuntimeError("YOLO returned more images than the classifier feature batch")
                probabilities = active_classifier_probabilities[active_classifier_index]
                active_classifier_index += 1
                classifier_class = max(range(len(probabilities)), key=probabilities.__getitem__)
                classifier_confidence = float(probabilities[classifier_class])
                labels_by_id[image_id] = classifier_class
                probabilities_by_id[image_id] = [float(v) for v in probabilities]

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                if classifier_class is None:
                    labels_by_id[image_id] = fallback_class
                continue

            xyxy = boxes.xyxy.detach().cpu().tolist()
            categories = boxes.cls.detach().cpu().tolist()
            scores = boxes.conf.detach().cpu().tolist()
            top_index = max(range(len(scores)), key=lambda index: float(scores[index]))
            # Submission format requires one image label even when no detection
            # clears the local metric's confidence threshold. Use its strongest box.
            if classifier_class is None:
                labels_by_id[image_id] = int(categories[top_index])

            scale_x = float(image["width"]) / IMAGE_SIZE
            scale_y = float(image["height"]) / IMAGE_SIZE
            for (x1, y1, x2, y2), class_index, score in zip(xyxy, categories, scores):
                x1, x2 = sorted((max(0.0, min(float(IMAGE_SIZE), float(x1))),
                                 max(0.0, min(float(IMAGE_SIZE), float(x2)))) )
                y1, y2 = sorted((max(0.0, min(float(IMAGE_SIZE), float(y1))),
                                 max(0.0, min(float(IMAGE_SIZE), float(y2)))) )
                left, right = x1 * scale_x, x2 * scale_x
                top, bottom = y1 * scale_y, y2 * scale_y
                if right <= left or bottom <= top:
                    continue
                if args.box_class_policy != "native_detector_classes":
                    class_index = classifier_class
                    if args.box_class_policy == "classifier_class_score_weighted":
                        score = float(score) * classifier_confidence
                detections.append({
                    "image_id": image_id,
                    "category_id": int(class_index) + 1,
                    "bbox": [left, top, right - left, bottom - top],
                    "score": float(score),
                })

        if processed != len(images):
            raise RuntimeError(f"Predicted {processed}/{len(images)} official validation images")
        actual_branch = "one2one" if getattr(detector.predictor.model, "end2end", False) else "one2many"
        if args.detection_branch != "default" and actual_branch != args.detection_branch:
            raise RuntimeError(f"Requested {args.detection_branch}, but the inference backend selected {actual_branch}")
        if classifier_hook is not None:
            classifier_hook.remove()

    if set(labels_by_id) != {int(image["id"]) for image in images}:
        raise RuntimeError("Submission is missing one or more image-level labels")
    elapsed = time.perf_counter() - start
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        with archive.open("classification.csv", "w") as raw:
            import io

            text_stream = io.TextIOWrapper(raw, encoding="utf-8", newline="", write_through=True)
            writer = csv.writer(text_stream)
            writer.writerow(["image_id", "label"])
            for image in images:
                image_id = int(image["id"])
                writer.writerow([image_id, labels_by_id[image_id]])
            text_stream.flush()
            text_stream.detach()
        archive.writestr("detection.json", json.dumps(detections, separators=(",", ":")))
    temporary_zip.replace(args.out)
    if probabilities_by_id:
        args.out.with_suffix('.probabilities.json').write_text(json.dumps(probabilities_by_id))

    metadata = {
        "format": "project classification.csv + COCO detection.json",
        "split": "course validation; unlabeled leaderboard split",
        "images": len(images),
        "detections": len(detections),
        "image_size": IMAGE_SIZE,
        "input_channels": "grayscale; Ultralytics expands one channel to three identical channels",
        "resize": "PIL bilinear stretch to 512 x 512; boxes mapped back to original image coordinates",
        "input_filter": input_filter,
        "classification_rule": (
            "supervised image-classification head top-1 class"
            if classifier_head is not None else
            "highest-confidence detection class; if no detections, use the most frequent image class "
            "in course training labels"
        ),
        "box_class_policy": args.box_class_policy,
        "detection_branch": args.detection_branch,
        "effective_detection_branch": actual_branch,
        "nms_iou": args.nms_iou,
        "multi_label": args.multi_label,
        "classifier_feature_pooling": classifier_payload.get("feature_pooling", "mean") if classifier_head is not None else None,
        "classifier_checkpoint": str(args.classifier_checkpoint) if args.classifier_checkpoint else None,
        "classification_fallback_class": fallback_class,
        "batch_size": args.batch,
        "inference_seconds_including_resize_and_zip": elapsed,
        "weights": str(args.weights),
        "weights_sha256": sha256(args.weights),
        "zip": str(args.out),
        "zip_sha256": sha256(args.out),
    }
    args.out.with_suffix(".inference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.out}: {len(images)} labels, {len(detections)} boxes", flush=True)


if __name__ == "__main__":
    main()
