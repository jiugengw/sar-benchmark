#!/usr/bin/env python3
"""Train a scale of YOLO26 with a new stride-4 P2 detection level."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parameter_fingerprint(model) -> str:
    """Fingerprint parameter values before any optimizer step, independent of device."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def preserve_initialization(model, audit_path: Path) -> dict:
    """Register transferred weights with the wrapper and verify the trainer receives them.

    Layer state_dict loads alone leave a YAML wrapper without a checkpoint marker.
    The public load() API marks our prepared model as the initialization. Training
    must also use pretrained=True, or the wrapper explicitly discards those values.
    """
    expected = parameter_fingerprint(model.model)
    model.load(model.model)
    if not model.ckpt or parameter_fingerprint(model.model) != expected:
        raise RuntimeError('Could not register the transferred P2 model as training initialization')
    audit = {'prepared_parameters_sha256': expected, 'training_handoff_verified': False}
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')

    def verify(trainer):
        actual = parameter_fingerprint(trainer.model)
        audit.update(trainer_parameters_sha256=actual, training_handoff_verified=actual == expected)
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
        if actual != expected:
            raise RuntimeError('Trainer discarded or changed initialization before the first optimizer step')
        print(f'INITIALIZATION_HANDOFF_VERIFIED {actual}', flush=True)

    model.add_callback('on_pretrain_routine_end', verify)
    return audit


def transfer_standard_weights_to_p2(standard_model, p2_model) -> None:
    """Transfer semantically matching YOLO26 layers and P3-P5 detection heads."""
    source = standard_model.model
    target = p2_model.model

    # The new P2 branch occupies target layers 17-22, shifting the P4/P5
    # neck layers six positions later than in the standard YOLO26 model.
    layer_pairs = [(index, index) for index in range(17)]
    layer_pairs.extend((source_index, source_index + 6) for source_index in range(17, 23))
    for source_index, target_index in layer_pairs:
        source_layer = source.model[source_index]
        target_layer = target.model[target_index]
        source_state = source_layer.state_dict()
        target_state = target_layer.state_dict()
        transferable = {
            key: value for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        target_layer.load_state_dict(transferable, strict=False)

    source_detect = source.model[-1]
    target_detect = target.model[-1]
    for attribute in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        source_layers = getattr(source_detect, attribute, None)
        target_layers = getattr(target_detect, attribute, None)
        if source_layers is None or target_layers is None:
            continue
        if len(source_layers) != 3 or len(target_layers) != 4:
            raise RuntimeError(
                f"Unexpected {attribute} levels: {len(source_layers)} -> {len(target_layers)}"
            )
        for source_index, target_index in enumerate((1, 2, 3)):
            source_module = source_layers[source_index]
            target_module = target_layers[target_index]
            source_state = source_module.state_dict()
            target_state = target_module.state_dict()
            transferable = {
                key: value for key, value in source_state.items()
                if key in target_state and target_state[key].shape == value.shape
            }
            target_module.load_state_dict(transferable, strict=False)

    if hasattr(source_detect, "dfl") and hasattr(target_detect, "dfl"):
        target_detect.dfl.load_state_dict(source_detect.dfl.state_dict(), strict=False)


def build_p2_model(source, scale, p2_yaml, detect_yaml, class_names, output):
    """YOLO26-P2 for our classes, initialised from a pretrained YOLO26 of any task.

    A detection checkpoint (COCO) is first loaded into the standard detection model. Another task's checkpoint, such
    as yolo26m-obb.pt (rotated boxes on DOTA aerial photos), goes into the standard detection architecture instead:
    every layer with the same name and shape transfers (backbone, neck, box branches); its class and angle outputs do
    not. The P2 model then takes the shared layers and the P3-P5 heads, as before."""
    from ultralytics import YOLO

    if source.task == "detect":
        standard_config = copy.deepcopy(source.model.yaml)
    else:
        standard_config = yaml.safe_load(detect_yaml.read_text())
        standard_config["scale"] = scale
    standard_config["nc"] = len(class_names)
    standard_config_path = output / f"yolo26{scale}_9class.yaml"
    standard_config_path.write_text(yaml.safe_dump(standard_config, sort_keys=False))
    standard_model = YOLO(str(standard_config_path), task="detect")
    standard_model.model.names = dict(enumerate(class_names))
    standard_model.model.load(source.model, verbose=True)

    p2_config = yaml.safe_load(p2_yaml.read_text())
    p2_config["nc"] = len(class_names)
    p2_config["scale"] = scale
    p2_config_path = output / f"yolo26{scale}_p2.yaml"
    p2_config_path.write_text(yaml.safe_dump(p2_config, sort_keys=False))
    model = YOLO(str(p2_config_path), task="detect")
    model.model.names = dict(enumerate(class_names))
    transfer_standard_weights_to_p2(standard_model, model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("n", "s", "m", "l"), default="s")
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--p2-yaml",
        type=Path,
        default=Path(".venv/lib/python3.12/site-packages/ultralytics/cfg/models/26/yolo26-p2.yaml"),
    )
    parser.add_argument(
        "--detect-yaml",
        type=Path,
        default=Path(".venv/lib/python3.12/site-packages/ultralytics/cfg/models/26/yolo26.yaml"),
        help="detection architecture used when --weights is another task, e.g. yolo26m-obb.pt (DOTA aerial photos)",
    )
    parser.add_argument("--classes", type=Path, default=Path("data/yolo11s-sar-512/conversion_manifest.json"))
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--flipud", type=float, default=0.0,
                        help="vertical-flip probability; overhead images have no up, so 0.5 is a free augmentation")
    parser.add_argument("--close-mosaic", type=int, default=10, help="turn mosaic off for the last N epochs")
    args = parser.parse_args()
    if args.weights is None:
        args.weights = Path(f"pretrained/yolo26{args.scale}.pt")

    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("YOLO training requires the allocated CUDA GPU")
    if not args.p2_yaml.is_file():
        raise FileNotFoundError(args.p2_yaml)
    class_names = json.loads(args.classes.read_text())["classes"]
    if len(class_names) != 9:
        raise ValueError(f"Expected the nine original classes, found {class_names}")
    args.train_output.mkdir(parents=True, exist_ok=True)
    set_seed(0)

    if args.weights.is_file():
        source = YOLO(str(args.weights))
    elif args.scale == "n" and args.weights.name == "yolo26n.pt":
        # YOLO26n is available from Ultralytics but is not staged in this repo.
        # Resolve the official asset once and keep a local copy for evaluation.
        source = YOLO("yolo26n.pt")
        downloaded_weights = Path(source.ckpt_path)
        args.weights = args.train_output / "pretrained_yolo26n.pt"
        shutil.copy2(downloaded_weights, args.weights)
    else:
        raise FileNotFoundError(args.weights)
    model = build_p2_model(source, args.scale, args.p2_yaml, args.detect_yaml, class_names, args.train_output)
    source_first = source.model.model[0].state_dict()
    target_first = model.model.model[0].state_dict()
    if any(not torch.equal(value, target_first[key]) for key, value in source_first.items()):
        raise RuntimeError('Source backbone weights did not reach the P2 model')
    initialization = preserve_initialization(model, args.train_output / 'initialization_audit.json')
    source_task = source.task
    del source

    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    training_settings = {
        "variant": "p2",
        "architecture": f"YOLO26{args.scale}-P2 (stride 4/8/16/32)",
        "scale": args.scale,
        "pretrained_weights": str(args.weights),
        "pretrained_task": source_task,
        "pretrained_weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "initialization_handoff": initialization,
        "p2_architecture_yaml": str(args.p2_yaml),
        "classes": class_names,
        "epochs": args.epochs,
        "image_size": 512,
        "batch_size": args.batch,
        "workers": args.workers,
        "seed": 0,
        "flipud": args.flipud,
        "close_mosaic": args.close_mosaic,
        "optimizer": "auto (Ultralytics default)",
        "augmentation": "Ultralytics YOLO26 defaults",
        "trainable_parameters": parameter_count,
        "split": "runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json",
        "initialization": f"pretrained YOLO26{args.scale} ({source_task}) shared layers and P3-P5 heads; "
                          "new P2 branch initialized randomly",
    }
    (args.train_output / "benchmark_settings.json").write_text(json.dumps(training_settings, indent=2) + "\n")
    print(json.dumps(training_settings, indent=2), flush=True)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train(
        data="data/yolo11s-sar-512/dataset.yaml",
        epochs=args.epochs,
        imgsz=512,
        batch=args.batch,
        workers=args.workers,
        device=0,
        pretrained=True,
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
        flipud=args.flipud,
        close_mosaic=args.close_mosaic,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    compute = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_hours": elapsed / 3600.0,
        "wall_seconds": elapsed,
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated(0) / 1024**3,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(0) / 1024**3,
        "total_parameters": sum(parameter.numel() for parameter in model.model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.model.parameters()
                                     if parameter.requires_grad),
        "best_weights": str(args.train_output / "train/weights/best.pt"),
    }
    (args.train_output / "compute.json").write_text(json.dumps(compute, indent=2) + "\n")
    print(json.dumps(compute, indent=2), flush=True)


if __name__ == "__main__":
    main()
