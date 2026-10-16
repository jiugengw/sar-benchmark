#!/usr/bin/env python3
"""Train matched 100-epoch YOLO11s standard or P2 detection models."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_adapted_standard(weights: Path, class_names: list[str], config_path: Path):
    """Build the 9-class YOLO11s model and transfer its official pretrained weights."""
    from ultralytics import YOLO

    source = YOLO(str(weights))
    config = copy.deepcopy(source.model.yaml)
    config["nc"] = len(class_names)
    # Ultralytics treats a dict passed to its YOLO wrapper as a checkpoint
    # path, and the trainer reloads yaml_file during setup. Keep this adapted
    # YAML in the run directory for the full training lifetime.
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    model = YOLO(str(config_path), task="detect")
    model.model.names = dict(enumerate(class_names))
    model.model.load(source.model, verbose=True)
    return model


def transfer_standard_weights_to_p2(standard_model, p2_model) -> None:
    """Copy common backbone/neck weights and map pretrained P3-P5 heads to P2-P5."""
    source = standard_model.model
    target = p2_model.model
    source_state = source.state_dict()
    target_state = target.state_dict()
    shared = {
        key: value for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    target.load_state_dict(shared, strict=False)

    source_detect = source.model[-1]
    target_detect = target.model[-1]
    for attribute in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        source_layers = getattr(source_detect, attribute, None)
        target_layers = getattr(target_detect, attribute, None)
        if source_layers is None or target_layers is None:
            continue
        if len(source_layers) != 3 or len(target_layers) != 4:
            raise RuntimeError(f"Unexpected {attribute} levels: {len(source_layers)} -> {len(target_layers)}")
        # The P2 head is new; the stock P3/P4/P5 detection towers initialize
        # the matching target P3/P4/P5 towers (indices 1, 2, 3).
        for source_index, target_index in enumerate((1, 2, 3)):
            source_module = source_layers[source_index]
            target_module = target_layers[target_index]
            source_module_state = source_module.state_dict()
            target_module_state = target_module.state_dict()
            transferable = {
                key: value for key, value in source_module_state.items()
                if key in target_module_state and target_module_state[key].shape == value.shape
            }
            target_module.load_state_dict(transferable, strict=False)

    # DFL is shared by every level and has the same shape in both heads.
    if hasattr(source_detect, "dfl") and hasattr(target_detect, "dfl"):
        target_detect.dfl.load_state_dict(source_detect.dfl.state_dict(), strict=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("standard", "p2"), required=True)
    parser.add_argument("--weights", type=Path, default=Path("pretrained/yolo11s.pt"))
    parser.add_argument("--dataset-yaml", type=Path, default=Path("data/yolo11s-sar-512/dataset.yaml"))
    parser.add_argument("--classes", type=Path, default=Path("data/yolo11s-sar-512/conversion_manifest.json"))
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("YOLO training requires the allocated CUDA GPU")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    class_names = json.loads(args.classes.read_text())["classes"]
    if len(class_names) != 9:
        raise ValueError(f"Expected the nine original classes, found {class_names}")
    args.train_output.mkdir(parents=True, exist_ok=True)
    set_seed(0)

    standard_model = build_class_adapted_standard(
        args.weights, class_names, args.train_output / "yolo11s_9class.yaml"
    )
    if args.variant == "standard":
        model = standard_model
    else:
        p2_yaml = Path(__file__).with_name("yolo11s_p2.yaml")
        model = YOLO(str(p2_yaml), task="detect")
        model.model.names = dict(enumerate(class_names))
        transfer_standard_weights_to_p2(standard_model, model)
        del standard_model

    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    training_settings = {
        "variant": args.variant,
        "architecture": "YOLO11s-P2 (stride 4/8/16/32)" if args.variant == "p2" else "YOLO11s (stride 8/16/32)",
        "pretrained_weights": str(args.weights),
        "classes": class_names,
        "epochs": args.epochs,
        "image_size": 512,
        "batch_size": args.batch,
        "workers": args.workers,
        "seed": 0,
        "optimizer": "auto (Ultralytics default)",
        "augmentation": "identical Ultralytics YOLO11 defaults for both variants",
        "trainable_parameters": parameter_count,
        "split": "runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json",
    }
    (args.train_output / "benchmark_settings.json").write_text(json.dumps(training_settings, indent=2) + "\n")
    print(json.dumps(training_settings, indent=2), flush=True)

    model.train(
        data=str(args.dataset_yaml.resolve()),
        epochs=args.epochs,
        imgsz=512,
        batch=args.batch,
        workers=args.workers,
        device=0,
        pretrained=False,
        optimizer="auto",
        patience=args.epochs + 1,
        seed=0,
        deterministic=True,
        project=str(args.train_output.resolve()),
        name="train",
        exist_ok=False,
        plots=True,
        val=True,
        cache=False,
    )


if __name__ == "__main__":
    main()
