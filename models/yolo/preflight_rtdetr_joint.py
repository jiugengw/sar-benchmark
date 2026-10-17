#!/usr/bin/env python3
"""Check RT-DETR joint labels, gradient paths, validation features, and checkpoint loading."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from models.yolo.joint_rtdetr import JointRTDETRDetectionModel


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("models/yolo/experiment_configs/phase3_rtdetr_l.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from ultralytics import RTDETR
    from ultralytics.nn.tasks import load_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("RT-DETR joint preflight requires an allocated CUDA GPU")
    config = yaml.safe_load(resolve(args.config).read_text())
    experiment = config["experiments"]["E2"]
    detector_path = resolve(experiment["detector_checkpoint"])
    classifier_path = resolve(experiment["classifier_checkpoint"])
    dataset_root = resolve(config["dataset_root"])
    data_root = resolve(config["data_root"])
    split_path = resolve(config["split"])
    class_names = json.loads(resolve(config["classes"]).read_text())["classes"]
    split = json.loads(split_path.read_text())
    image_map = json.loads((dataset_root / "image_map.json").read_text())
    info_by_id = {int(info["image_id"]): (relative, info) for relative, info in image_map.items()}
    with (data_root / "train" / "labels.csv").open(newline="") as stream:
        labels_by_id = {int(row["image_id"]): int(row["label"]) for row in csv.DictReader(stream)}
    train_ids = list(map(int, split["train_image_ids"]))
    counts = torch.bincount(torch.tensor([labels_by_id[image_id] for image_id in train_ids]), minlength=len(class_names)).float()
    if (counts == 0).any():
        raise ValueError("Saved training split is missing one or more image classes")
    class_weights = (counts.sum() / (len(class_names) * counts)).tolist()

    source = RTDETR(str(detector_path))
    model = JointRTDETRDetectionModel(
        cfg=source.model.yaml,
        nc=len(class_names),
        ch=3,
        verbose=False,
        class_weights=class_weights,
        classification_loss_weight=float(config["classification_loss_weight"]),
    )
    model.load(source.model, verbose=False)
    classifier_payload = torch.load(classifier_path, map_location="cpu", weights_only=True)
    if classifier_payload["class_names"] != class_names:
        raise ValueError("E7 classifier class order differs from the converted dataset")
    if int(classifier_payload["feature_dim"]) != model.image_classifier.network[-1].in_features:
        raise ValueError("E7 classifier feature size does not match the RT-DETR decoder inputs")
    model.image_classifier.load_state_dict(classifier_payload["state_dict"], strict=True)

    images = []
    image_targets = []
    target_classes = []
    target_boxes = []
    target_batch_indices = []
    for batch_index, image_id in enumerate(train_ids):
        relative, info = info_by_id[image_id]
        image_path = dataset_root / relative
        label_path = dataset_root / relative.replace("images/", "labels/", 1)
        label_path = label_path.with_suffix(".txt")
        with Image.open(image_path) as image:
            image_array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        images.append(torch.from_numpy(image_array).permute(2, 0, 1).float().div_(255.0))
        image_targets.append(labels_by_id[image_id])
        label_rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
        if not label_rows:
            continue
        for row in label_rows:
            target_classes.append(int(row[0]))
            target_boxes.append([float(value) for value in row[1:5]])
            target_batch_indices.append(batch_index)
        if len(images) == 2:
            break
    if len(images) != 2 or len(target_boxes) < 2:
        raise RuntimeError("Could not assemble two labeled RT-DETR training samples from the converted split")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model.to(device).train()
    batch = {
        "img": torch.stack(images).to(device),
        "cls": torch.tensor(target_classes, dtype=torch.float32, device=device).reshape(-1, 1),
        "bboxes": torch.tensor(target_boxes, dtype=torch.float32, device=device),
        "batch_idx": torch.tensor(target_batch_indices, dtype=torch.float32, device=device),
        "sar_image_class": torch.tensor(image_targets, dtype=torch.long, device=device),
        "sar_image_class_valid": torch.ones(2, dtype=torch.bool, device=device),
    }
    total_loss, loss_items = model(batch)
    shared_parameter = next(parameter for name, parameter in model.named_parameters()
                            if name.startswith("model.0.") and parameter.ndim > 1)
    decoder_parameter = next(model.model[-1].parameters())
    classifier_parameter = next(model.image_classifier.parameters())
    detection_shared_grad = torch.autograd.grad(
        model._joint_detection_loss, shared_parameter, retain_graph=True, allow_unused=True
    )[0]
    image_shared_grad = torch.autograd.grad(
        model._joint_classification_loss, shared_parameter, retain_graph=True, allow_unused=True
    )[0]
    detection_decoder_grad = torch.autograd.grad(
        model._joint_detection_loss, decoder_parameter, retain_graph=True, allow_unused=True
    )[0]
    image_head_grad = torch.autograd.grad(
        model._joint_classification_loss, classifier_parameter, retain_graph=True, allow_unused=True
    )[0]
    norms = {
        "detection_to_shared_backbone": float(detection_shared_grad.norm()) if detection_shared_grad is not None else 0.0,
        "image_classification_to_shared_backbone": float(image_shared_grad.norm()) if image_shared_grad is not None else 0.0,
        "detection_to_decoder": float(detection_decoder_grad.norm()) if detection_decoder_grad is not None else 0.0,
        "image_classification_to_image_head": float(image_head_grad.norm()) if image_head_grad is not None else 0.0,
    }
    if any(value <= 0 for value in norms.values()):
        raise RuntimeError(f"A required joint-training gradient path is missing: {norms}")
    total_loss.backward()

    model.eval()
    with torch.inference_mode():
        model(batch["img"])
        feature_maps = model._joint_feature_maps
        model._joint_feature_maps = None
        if feature_maps is None:
            raise RuntimeError("RT-DETR validation forward did not capture decoder feature maps")
        validation_logits = model.image_classifier(
            torch.cat([feature.float().mean(dim=(2, 3)) for feature in feature_maps], dim=1)
        )
    if validation_logits.shape != (2, len(class_names)):
        raise RuntimeError(f"Unexpected validation classifier output shape: {tuple(validation_logits.shape)}")

    # Exercise Ultralytics' real checkpoint loader with the custom model class;
    # the full training runner uses this same loader after saving best.pt.
    model.criterion = None
    model._joint_detection_loss = None
    model._joint_classification_loss = None
    model.cpu().half().eval()
    with tempfile.TemporaryDirectory(prefix="phase3-rtdetr-preflight-") as temp_dir:
        checkpoint_path = Path(temp_dir) / "joint-rtdetr-preflight.pt"
        torch.save({"ema": model, "train_args": {}}, checkpoint_path)
        reloaded, _checkpoint = load_checkpoint(checkpoint_path, device="cpu")
        if not isinstance(reloaded, JointRTDETRDetectionModel) or not hasattr(reloaded, "image_classifier"):
            raise RuntimeError("Ultralytics checkpoint reload did not preserve the joint RT-DETR model")

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "passed",
        "samples": 2,
        "detection_boxes": len(target_boxes),
        "image_labels": image_targets,
        "class_names": class_names,
        "loss_items": {name: float(value) for name, value in loss_items.items()},
        "combined_loss": float(total_loss.detach()),
        "gradient_l2_norms": norms,
        "validation_classifier_shape": list(validation_logits.shape),
        "checkpoint_reload_class": type(reloaded).__name__,
        "checkpoint_reload_kept_image_classifier": True,
        "checkpoint_feature_dim": int(reloaded.image_classifier.network[-1].in_features),
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
