#!/usr/bin/env python3
"""Small-batch CPU/GPU smoke check for the E2 joint model and gradients."""

from __future__ import annotations

import argparse
import csv
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import IterableSimpleNamespace

from models.yolo.image_head import classification_metrics, pool_detection_features
from models.yolo.joint_model import JointDetectionModel
from models.yolo.joint_trainer import align_image_labels, combined_delta_m


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, default=Path("runs/yolo11s-100-p2-328058/train/weights/best.pt"))
    parser.add_argument("--classifier", type=Path, default=Path("runs/yolo11s-100-p2-328058/image_classifier.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/course-data"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/yolo11s-sar-512"))
    parser.add_argument("--split", type=Path, default=Path("runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    detector_path = args.detector if args.detector.is_absolute() else ROOT / args.detector
    classifier_path = args.classifier if args.classifier.is_absolute() else ROOT / args.classifier
    data_root = args.data if args.data.is_absolute() else ROOT / args.data
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else ROOT / args.dataset_root
    split_path = args.split if args.split.is_absolute() else ROOT / args.split
    for path in (detector_path, classifier_path, data_root, dataset_root, split_path):
        if not path.exists():
            raise FileNotFoundError(path)

    torch.manual_seed(0)
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = json.loads((dataset_root / "conversion_manifest.json").read_text())["classes"]
    split = json.loads(split_path.read_text())
    train_ids = list(map(int, split["train_image_ids"]))[: args.batch_size]
    with (data_root / "train" / "labels.csv").open(newline="") as stream:
        labels_by_id = {int(row["image_id"]): int(row["label"]) for row in csv.DictReader(stream)}
    image_map = json.loads((dataset_root / "image_map.json").read_text())
    image_by_id = {int(info["image_id"]): (relative, info) for relative, info in image_map.items()}
    class_by_stem = {Path(relative).stem: int(info["csv_image_label"]) for relative, info in image_map.items()}
    actual_paths = [str((dataset_root / image_by_id[image_id][0]).resolve()) for image_id in train_ids]
    aligned_targets, valid = align_image_labels(actual_paths, class_by_stem)
    if not valid.all() or aligned_targets.tolist() != [labels_by_id[image_id] for image_id in train_ids]:
        raise AssertionError("Image-level class labels do not align with the selected image paths")

    feature_res = args.image_size // 4
    images = []
    batch_indices = []
    boxes = []
    box_classes = []
    for index, image_id in enumerate(train_ids):
        relative, _info = image_by_id[image_id]
        image_path = dataset_root / relative
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize(
                (args.image_size, args.image_size), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(rgb, dtype=np.uint8).copy()
        images.append(torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0))
        label_path = dataset_root / "labels" / "train" / f"{Path(relative).stem}.txt"
        for line in label_path.read_text().splitlines():
            category, cx, cy, width, height = map(float, line.split())
            batch_indices.append(index)
            box_classes.append([category])
            boxes.append([cx, cy, width, height])

    batch = {
        "img": torch.stack(images).to(device),
        "batch_idx": torch.tensor(batch_indices, dtype=torch.float32, device=device),
        "cls": torch.tensor(box_classes, dtype=torch.float32, device=device),
        "bboxes": torch.tensor(boxes, dtype=torch.float32, device=device),
        "sar_image_class": aligned_targets.to(device),
        "sar_image_class_valid": valid.to(device),
    }

    source = YOLO(str(detector_path))
    source_model = source.model
    class_counts = torch.bincount(
        torch.tensor([labels_by_id[image_id] for image_id in split["train_image_ids"]]),
        minlength=len(class_names),
    ).float()
    class_weights = (class_counts.sum() / (len(class_names) * class_counts)).tolist()
    model = JointDetectionModel(
        cfg=source_model.yaml,
        nc=len(class_names),
        ch=3,
        verbose=False,
        class_weights=class_weights,
        classification_loss_weight=0.5,
    )
    model.names = dict(enumerate(class_names))
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, **source_model.args})
    model.load(source_model, verbose=False)
    classifier_payload = torch.load(classifier_path, map_location="cpu", weights_only=True)
    model.image_classifier.load_state_dict(classifier_payload["state_dict"], strict=True)
    model.to(device).train()

    # Check that both losses independently backpropagate into a shared backbone tensor.
    shared_parameter = model.model[0].conv.weight
    detector_loss, _ = DetectionModel.loss(model, batch)
    detector_loss.sum().backward()
    detection_gradient = float(shared_parameter.grad.abs().sum()) if shared_parameter.grad is not None else 0.0
    model.zero_grad(set_to_none=True)

    model(batch["img"])
    if model._joint_feature_maps is None:
        raise AssertionError("Detect pre-hook captured no feature maps")
    shapes = [tuple(feature.shape[-2:]) for feature in model._joint_feature_maps]
    expected_shapes = [(feature_res, feature_res), (feature_res // 2, feature_res // 2),
                       (feature_res // 4, feature_res // 4), (feature_res // 8, feature_res // 8)]
    if shapes != expected_shapes:
        raise AssertionError(f"Expected stride-4/8/16/32 feature maps {expected_shapes}; found {shapes}")
    classifier_logits = model.image_classifier(pool_detection_features(model._joint_feature_maps))
    classifier_loss = F.cross_entropy(
        classifier_logits,
        batch["sar_image_class"],
        weight=model.image_class_weights,
    )
    classifier_loss.backward()
    classification_gradient = float(shared_parameter.grad.abs().sum()) if shared_parameter.grad is not None else 0.0
    if detection_gradient <= 0 or classification_gradient <= 0:
        raise AssertionError(
            f"Shared backbone gradients are missing: detection={detection_gradient}, classification={classification_gradient}"
        )

    model.zero_grad(set_to_none=True)
    total_loss, loss_items = model.loss(batch)
    total_loss.sum().backward()
    detector_head_parameter = model.model[-1].cv2[0][0].conv.weight
    gradient_summary = {
        "shared_backbone_from_detection_loss": detection_gradient,
        "shared_backbone_from_classification_loss": classification_gradient,
        "detection_head_from_total_loss": float(detector_head_parameter.grad.abs().sum()),
        "image_classifier_from_total_loss": float(model.image_classifier.network[-1].weight.grad.abs().sum()),
    }
    if any(value <= 0 for value in gradient_summary.values()):
        raise AssertionError(f"Expected non-zero gradients from both losses: {gradient_summary}")

    model._joint_feature_maps = None
    ema_copy = copy.deepcopy(model).eval()
    ema_copy(batch["img"])
    if ema_copy._joint_feature_maps is None or model._joint_feature_maps is not None:
        raise AssertionError("The Detect feature hook did not follow the EMA model copy")

    # Ultralytics' detector validator passes predictions but does not attach
    # the image labels that the training preprocessor adds. Confirm this path
    # returns detection-only loss; the trainer validates classifier F1 using
    # its separate labeled loader.
    model.eval()
    validation_batch = {
        key: value for key, value in batch.items()
        if key not in {"sar_image_class", "sar_image_class_valid"}
    }
    with torch.inference_mode():
        validation_predictions = model(batch["img"])
        _validation_loss, validation_items = model.loss(validation_batch, validation_predictions)
    if float(validation_items["image_cls_loss"]) != 0.0:
        raise AssertionError("Detector-only validation should not include image-classification loss")
    model.train()

    predictions = classifier_logits.argmax(dim=1).cpu().tolist()
    cls_metrics = classification_metrics(aligned_targets.tolist(), predictions, class_names)
    delta = combined_delta_m(cls_metrics["macro_f1"], 0.3)
    with tempfile.TemporaryDirectory(prefix="sar-joint-smoke-") as temp_dir:
        checkpoint_path = Path(temp_dir) / "joint_state.pt"
        torch.save(model.state_dict(), checkpoint_path)
        restored = JointDetectionModel(
            cfg=source_model.yaml,
            nc=len(class_names),
            ch=3,
            verbose=False,
            class_weights=class_weights,
            classification_loss_weight=0.5,
        )
        restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)

    print(json.dumps({
        "status": "passed",
        "device": str(device),
        "batch_size": len(train_ids),
        "image_size": args.image_size,
        "aligned_class_targets": aligned_targets.tolist(),
        "detection_boxes": len(boxes),
        "feature_map_shapes": shapes,
        "joint_loss_items": {key: float(value.detach().cpu()) for key, value in loss_items.items()},
        "gradient_summary": gradient_summary,
        "macro_f1_on_smoke_examples": cls_metrics["macro_f1"],
        "combined_delta_m_example": delta,
        "state_dict_reload": "passed",
        "ema_copy_feature_hook": "passed",
        "detector_validation_without_image_labels": "passed",
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
