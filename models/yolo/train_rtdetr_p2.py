#!/usr/bin/env python3
"""Preflight stock RT-DETR-L, then fine-tune an RT-DETR-L model with P2."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from rtdetr_support import RTDETRP2Trainer, transfer_standard_rtdetr_to_p2


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def standard_preflight(checkpoint: Path, image_path: Path, output: Path, device: torch.device) -> None:
    """Run a stock pretrained forward pass and record output/weight facts first."""
    from PIL import Image
    from ultralytics import RTDETR

    model = RTDETR(str(checkpoint))
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
    model.model.to(device).eval()
    with torch.inference_mode():
        predictions = model.model(tensor.to(device))
    summary = {
        "status": "passed",
        "architecture": "stock pretrained RT-DETR-L before P2 changes",
        "checkpoint": str(checkpoint),
        "checkpoint_classes": len(model.names),
        "input_shape": list(tensor.shape),
        "output_tensor_shapes": [list(value.shape) for value in predictions if isinstance(value, torch.Tensor)]
        if isinstance(predictions, (list, tuple)) else [list(predictions.shape)],
        "validation_scope": "pretrained architecture forward-pass preflight; SAR P2 benchmark training follows",
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("pretrained/rtdetr-l.pt"))
    parser.add_argument("--p2-yaml", type=Path, default=Path("models/yolo/rtdetr_l_p2.yaml"))
    parser.add_argument("--dataset-yaml", type=Path, default=Path("data/yolo11s-sar-512/dataset.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/yolo11s-sar-512"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    from ultralytics import RTDETR, __version__ as ultralytics_version

    weights = resolve(args.weights)
    output = resolve(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not weights.is_file():
        # Ultralytics resolves its official checkpoint URL; keep the result staged in the project.
        staged = RTDETR("rtdetr-l.pt")
        source_path = Path(staged.ckpt_path)
        if not source_path.is_file():
            raise FileNotFoundError("Ultralytics did not stage the official rtdetr-l.pt checkpoint")
        weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, weights)
        del staged
    yaml_path = resolve(args.p2_yaml)
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    if not torch.cuda.is_available():
        raise RuntimeError("RT-DETR training requires the allocated CUDA GPU")

    device = torch.device("cuda:0")
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(args.workers)
    preflight_image = resolve(args.dataset_root) / "images/train/00001.png"
    standard_preflight(weights, preflight_image, output / "standard_preflight.json", device)

    # Independently verify weight remapping, P2 input use, and resulting feature dimensions.
    source = RTDETR(str(weights))
    target = RTDETR(str(yaml_path))
    target.model.names = source.model.names
    transfer_summary = transfer_standard_rtdetr_to_p2(source.model, target.model)
    captured = []
    handle = target.model.model[-1].register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0]))
    target.model.to(device).eval()
    with torch.inference_mode():
        target.model(torch.zeros((1, 3, 256, 256), device=device))
    handle.remove()
    feature_maps = captured[0]
    feature_shapes = [list(feature.shape[-2:]) for feature in feature_maps]
    expected = [[64, 64], [32, 32], [16, 16], [8, 8]]
    if feature_shapes != expected:
        raise RuntimeError(f"RT-DETR P2 decoder inputs are {feature_shapes}, expected stride-4/8/16/32 {expected}")
    names = json.loads((resolve(args.dataset_root) / "conversion_manifest.json").read_text())["classes"]
    del source, target, feature_maps
    torch.cuda.empty_cache()

    from ultralytics import RTDETR as RTDETRModel

    model = RTDETRModel(str(yaml_path))
    settings = {
        "experiment": "E7",
        "architecture": "RT-DETR-L-P2",
        "pretrained_checkpoint": str(weights),
        "pretrained_checkpoint_sha256": __import__("hashlib").sha256(weights.read_bytes()).hexdigest(),
        "p2_yaml": str(yaml_path),
        "dataset_yaml": str(resolve(args.dataset_yaml)),
        "classes": names,
        "epochs": args.epochs,
        "image_size": 512,
        "batch_size": args.batch,
        "optimizer": "AdamW, lr0=0.0001, lrf=0.1, cosine schedule",
        "augmentation": "Ultralytics RT-DETR defaults; mosaic/mixup/copy-paste disabled",
        "seed": seed,
        "deterministic": False,
        "transfer": transfer_summary,
        "decoder_feature_shapes_at_256": feature_shapes,
        "ultralytics_version": ultralytics_version,
    }
    (output / "experiment_config.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(settings, indent=2), flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model.train(
        trainer=RTDETRP2Trainer,
        data=str(resolve(args.dataset_yaml)),
        epochs=args.epochs,
        imgsz=512,
        batch=args.batch,
        workers=args.workers,
        device=0,
        pretrained=str(weights),
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=1.0,
        weight_decay=1e-4,
        patience=args.epochs + 1,
        seed=seed,
        deterministic=False,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=0,
        project=str(output),
        name="train",
        exist_ok=False,
        plots=True,
        val=True,
        cache=False,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    best = output / "train/weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"RT-DETR P2 training did not produce {best}")
    compute = {
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_hours": elapsed / 3600.0,
        "wall_seconds": elapsed,
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "total_parameters": sum(parameter.numel() for parameter in model.model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.model.parameters() if parameter.requires_grad),
        "best_weights": str(best),
    }
    (output / "compute.json").write_text(json.dumps(compute, indent=2) + "\n")
    print(json.dumps(compute, indent=2), flush=True)


if __name__ == "__main__":
    main()
