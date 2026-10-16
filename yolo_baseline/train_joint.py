#!/usr/bin/env python3
"""Run a configurable joint YOLO-P2 classification/detection experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from yolo_baseline.joint_trainer import JointDetectionTrainer


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("yolo_baseline/experiment_configs/phase1.yaml"))
    parser.add_argument("--experiment", choices=("E2", "E3"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="unique run directory")
    args = parser.parse_args()

    from ultralytics import YOLO, __version__ as ultralytics_version

    if not torch.cuda.is_available():
        raise RuntimeError("Joint YOLO training requires the allocated CUDA GPU")
    config_path = resolve(str(args.config))
    config = yaml.safe_load(config_path.read_text())
    experiment = config["experiments"][args.experiment]
    output = resolve(str(args.output))
    if output.exists() and any(output.iterdir()):
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
    classifier_checkpoint = resolve(experiment.get("classifier_checkpoint"))
    detector_checkpoint = resolve(experiment.get("detector_checkpoint"))
    pretrained_weights = resolve(experiment.get("pretrained_weights"))

    if args.experiment == "E2":
        if not detector_checkpoint or not detector_checkpoint.is_file():
            raise FileNotFoundError(f"Missing E2 detector checkpoint: {detector_checkpoint}")
        if not classifier_checkpoint or not classifier_checkpoint.is_file():
            raise FileNotFoundError(f"Missing E2 classifier checkpoint: {classifier_checkpoint}")
        model = YOLO(str(detector_checkpoint))
        pretrained = True
    else:
        if not pretrained_weights or not pretrained_weights.is_file():
            raise FileNotFoundError(f"Missing E3 pretrained detector checkpoint: {pretrained_weights}")
        source_yaml = resolve(experiment["p2_model_yaml"])
        model_yaml = output / source_yaml.name
        p2_config = yaml.safe_load(source_yaml.read_text())
        model_yaml.write_text(yaml.safe_dump(p2_config, sort_keys=False))
        model = YOLO(str(model_yaml), task="detect")
        pretrained = str(pretrained_weights)

    JointDetectionTrainer.configure(
        dataset_root=str(dataset_root),
        data_root=str(data_root),
        split_path=str(split_path),
        classes_path=str(classes_path),
        classifier_checkpoint=str(classifier_checkpoint) if classifier_checkpoint else None,
        architecture_family=str(config.get("architecture_family", "yolo11")),
        classification_loss_weight=float(config["classification_loss_weight"]),
        shared_lr_scale=float(config["shared_lr_scale"]),
    )

    train_args = {
        "data": str(dataset_yaml),
        "epochs": int(experiment["epochs"]),
        "imgsz": int(config["image_size"]),
        "batch": int(config["batch_size"]),
        "workers": int(config["workers"]),
        "device": 0,
        "pretrained": pretrained,
        "optimizer": str(config["optimizer"]),
        "lr0": float(config["lr0"]),
        "lrf": float(config["lrf"]),
        "warmup_bias_lr": 0.0,
        "weight_decay": float(config["weight_decay"]),
        "patience": int(experiment["epochs"]) + 1,
        "seed": seed,
        "deterministic": True,
        "project": str(output),
        "name": "train",
        "exist_ok": False,
        "plots": True,
        "val": True,
        "cache": False,
        # Mosaic and mixup combine distinct image classes and invalidate a single image label.
        "mosaic": float(config["augmentation"]["mosaic"]),
        "mixup": float(config["augmentation"]["mixup"]),
        "copy_paste": float(config["augmentation"]["copy_paste"]),
        "close_mosaic": 0,
        "multi_scale": 0.0,
    }
    settings = {
        "experiment": args.experiment,
        "name": experiment["name"],
        "strategy": experiment["strategy"],
        "architecture": str(config.get("architecture", "YOLO11s-P2")),
        "pretrained_weights": str(detector_checkpoint or pretrained_weights),
        "pretrained_weights_sha256": sha256(detector_checkpoint or pretrained_weights),
        "classification_checkpoint_initialization": str(classifier_checkpoint) if classifier_checkpoint else "random",
        "epochs": int(experiment["epochs"]),
        "image_size": int(config["image_size"]),
        "batch_size": int(config["batch_size"]),
        "seed": seed,
        "optimizer": config["optimizer"],
        "head_learning_rate": config["lr0"],
        "shared_backbone_neck_learning_rate": float(config["lr0"]) * float(config["shared_lr_scale"]),
        "classification_loss_weight": config["classification_loss_weight"],
        "augmentation": config["augmentation"],
        "split": str(split_path),
        "ultralytics_version": ultralytics_version,
    }
    (output / "experiment_config.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(settings, indent=2), flush=True)

    device = torch.device("cuda:0")
    # Initialize the allocated CUDA context before querying allocator peaks.
    # On the Slurm MIG device, reset_peak_memory_stats can otherwise fail when
    # no CUDA tensor has reached the device yet.
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model.train(trainer=JointDetectionTrainer, **train_args)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    best_weights = output / "train" / "weights" / "best.pt"
    if not best_weights.is_file() or not hasattr(model.model, "image_classifier"):
        raise RuntimeError("Joint training did not preserve the combined-metric best checkpoint and classifier")
    class_names = json.loads(classes_path.read_text())["classes"]
    classifier = model.model.image_classifier.cpu()
    classifier_payload = {
        "format": "sar-yolo-image-head-v1",
        "feature_dim": int(classifier.network[-1].in_features),
        "num_classes": len(class_names),
        "class_names": class_names,
        "dropout": 0.2,
        "state_dict": classifier.state_dict(),
        "feature_source": "global average pooling of every P2-P5 feature map passed to YOLO's final Detect module",
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
    per_class_path = output / "best_classifier_metrics.json"
    best_classifier_metrics = getattr(model.trainer, "best_classifier_metrics", None)
    if best_classifier_metrics is None:
        raise RuntimeError("Joint trainer did not retain classifier metrics for its best combined checkpoint")
    per_class_path.write_text(json.dumps(best_classifier_metrics, indent=2) + "\n")

    compute = {
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_hours": elapsed / 3600.0,
        "wall_seconds": elapsed,
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "total_parameters": sum(parameter.numel() for parameter in model.model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.model.parameters() if parameter.requires_grad),
        "best_combined_delta_m": model.trainer.best_combined_fitness,
        "best_weights": str(best_weights),
        "best_weights_sha256": sha256(best_weights),
        "classifier_checkpoint": str(classifier_path),
        "classifier_metrics": str(per_class_path),
    }
    (output / "compute.json").write_text(json.dumps(compute, indent=2) + "\n")
    print(json.dumps(compute, indent=2), flush=True)


if __name__ == "__main__":
    main()
