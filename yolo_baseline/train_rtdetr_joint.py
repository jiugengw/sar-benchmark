#!/usr/bin/env python3
"""Joint fine-tune the E7 RT-DETR-L-P2 detector and image classifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from yolo_baseline.joint_rtdetr import JointRTDETRTrainer


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("yolo_baseline/experiment_configs/phase3_rtdetr_l.yaml"))
    parser.add_argument("--experiment", choices=("E2",), default="E2")
    parser.add_argument("--output", type=Path, required=True, help="unique run directory")
    args = parser.parse_args()

    from ultralytics import RTDETR, __version__ as ultralytics_version

    if not torch.cuda.is_available():
        raise RuntimeError("Joint RT-DETR training requires the allocated CUDA GPU")
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text())
    experiment = config["experiments"][args.experiment]
    output = resolve(args.output)
    unexpected_outputs = [path for path in output.iterdir() if path.name != "preflight.json"] if output.exists() else []
    if unexpected_outputs:
        raise FileExistsError(f"Refusing to overwrite non-empty experiment directory {output}")
    output.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(config["workers"]))

    dataset_root = resolve(config["dataset_root"])
    data_root = resolve(config["data_root"])
    split_path = resolve(config["split"])
    classes_path = resolve(config["classes"])
    dataset_yaml = resolve(config["dataset_yaml"])
    detector_checkpoint = resolve(experiment["detector_checkpoint"])
    classifier_checkpoint = resolve(experiment["classifier_checkpoint"])
    for path in (dataset_root / "image_map.json", data_root / "train" / "labels.csv", split_path,
                 classes_path, dataset_yaml, detector_checkpoint, classifier_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Phase 3 input: {path}")

    JointRTDETRTrainer.configure(
        dataset_root=str(dataset_root),
        data_root=str(data_root),
        split_path=str(split_path),
        classes_path=str(classes_path),
        classifier_checkpoint=str(classifier_checkpoint),
        classification_loss_weight=float(config["classification_loss_weight"]),
        shared_lr_scale=float(config["shared_lr_scale"]),
        classifier_lr=float(config["classifier_lr"]),
    )

    settings = {
        "experiment": args.experiment,
        "name": experiment["name"],
        "strategy": experiment["strategy"],
        "architecture": config["architecture"],
        "detector_checkpoint": str(detector_checkpoint),
        "detector_checkpoint_sha256": sha256(detector_checkpoint),
        "classifier_checkpoint": str(classifier_checkpoint),
        "classifier_checkpoint_sha256": sha256(classifier_checkpoint),
        "epochs": int(experiment["epochs"]),
        "image_size": int(config["image_size"]),
        "batch_size": int(config["batch_size"]),
        "seed": seed,
        "optimizer": config["optimizer"],
        "detector_decoder_learning_rate": float(config["lr0"]),
        "shared_backbone_neck_learning_rate": float(config["lr0"]) * float(config["shared_lr_scale"]),
        "image_classifier_learning_rate": float(config["classifier_lr"]),
        "classification_loss_weight": float(config["classification_loss_weight"]),
        "augmentation": config["augmentation"],
        "split": str(split_path),
        "ultralytics_version": ultralytics_version,
        "amp": "Ultralytics default; matches E7 training settings",
    }
    (output / "experiment_config.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(settings, indent=2), flush=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = RTDETR(str(detector_checkpoint))
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model.train(
        trainer=JointRTDETRTrainer,
        data=str(dataset_yaml),
        epochs=int(experiment["epochs"]),
        imgsz=int(config["image_size"]),
        batch=int(config["batch_size"]),
        workers=int(config["workers"]),
        device=0,
        pretrained=True,
        amp=bool(config.get("amp", True)),
        optimizer=str(config["optimizer"]),
        lr0=float(config["lr0"]),
        lrf=float(config["lrf"]),
        cos_lr=True,
        warmup_epochs=1.0,
        warmup_bias_lr=0.0,
        weight_decay=float(config["weight_decay"]),
        patience=int(experiment["epochs"]) + 1,
        seed=seed,
        deterministic=False,
        mosaic=float(config["augmentation"]["mosaic"]),
        mixup=float(config["augmentation"]["mixup"]),
        copy_paste=float(config["augmentation"]["copy_paste"]),
        close_mosaic=0,
        multi_scale=0.0,
        project=str(output),
        name="train",
        exist_ok=False,
        plots=True,
        val=True,
        cache=False,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    best_weights = output / "train" / "weights" / "best.pt"
    if not best_weights.is_file() or not hasattr(model.model, "image_classifier"):
        raise RuntimeError("Joint RT-DETR training did not preserve the best combined checkpoint and classifier")
    class_names = json.loads(classes_path.read_text())["classes"]
    classifier = model.model.image_classifier.cpu()
    classifier_payload = {
        "format": "sar-rtdetr-image-head-v1",
        "feature_dim": int(classifier.network[-1].in_features),
        "num_classes": len(class_names),
        "class_names": class_names,
        "dropout": 0.2,
        "state_dict": classifier.state_dict(),
        "feature_source": "global average pooling of each P2-P5 feature map passed to RT-DETR-L's decoder",
        "supervision": "jointly trained from labels.csv using the saved 8453/939 split",
        "architecture": "LayerNorm + Dropout(0.2) + Linear(feature_dim, 9)",
        "training": {
            "strategy": experiment["strategy"],
            "epochs": int(experiment["epochs"]),
            "checkpoint_selection": "best local validation combined delta_m",
            "classification_loss_weight": config["classification_loss_weight"],
            "seed": seed,
        },
    }
    classifier_path = output / "image_classifier.pt"
    torch.save(classifier_payload, classifier_path)
    trainer = model.trainer
    best_classifier_metrics = trainer.best_classifier_metrics
    if best_classifier_metrics is None:
        raise RuntimeError("Joint RT-DETR trainer did not retain image-classification metrics for its best checkpoint")
    metrics_path = output / "best_classifier_metrics.json"
    metrics_path.write_text(json.dumps(best_classifier_metrics, indent=2) + "\n")
    best_detection_metrics = trainer.best_detection_metrics
    if best_detection_metrics is None:
        raise RuntimeError("Joint RT-DETR trainer did not retain detector per-class metrics for its best checkpoint")
    detection_metrics_path = output / "best_detection_metrics.json"
    detection_metrics_path.write_text(json.dumps(best_detection_metrics, indent=2) + "\n")

    compute = {
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_hours": elapsed / 3600.0,
        "wall_seconds": elapsed,
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "total_parameters": sum(parameter.numel() for parameter in model.model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.model.parameters() if parameter.requires_grad),
        "best_combined_delta_m": trainer.best_combined_fitness,
        "best_weights": str(best_weights),
        "best_weights_sha256": sha256(best_weights),
        "classifier_checkpoint": str(classifier_path),
        "classifier_metrics": str(metrics_path),
        "detection_metrics": str(detection_metrics_path),
        "loaded_checkpoint_class": type(model.model).__name__,
        "loaded_classifier_feature_dim": int(model.model.image_classifier.network[-1].in_features),
    }
    (output / "compute.json").write_text(json.dumps(compute, indent=2) + "\n")
    print(json.dumps(compute, indent=2), flush=True)


if __name__ == "__main__":
    main()
