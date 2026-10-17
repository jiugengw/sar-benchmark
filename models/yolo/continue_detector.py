#!/usr/bin/env python3
"""Conditionally resume E1's optimizer state through 200 total detector epochs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def continuation_gate(config: dict) -> dict:
    gate = config["continuation_gate"]
    path = resolve(gate["source_results_csv"])
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    metric_key = gate["metric"]
    window = int(gate["recent_window_epochs"])
    if len(rows) < window:
        return {"eligible": False, "reason": f"need {window} epochs, found {len(rows)}", "source": str(path)}
    recent = rows[-window:]
    start_value = float(recent[0][metric_key])
    end_value = float(recent[-1][metric_key])
    improvement = end_value - start_value
    required = float(gate["minimum_improvement_percentage_points"]) / 100.0
    return {
        "eligible": improvement >= required,
        "reason": "mAP trend passes threshold" if improvement >= required else "mAP trend is below threshold",
        "source": str(path),
        "metric": metric_key,
        "window_epochs": window,
        "first_epoch": int(recent[0]["epoch"]),
        "last_epoch": int(recent[-1]["epoch"]),
        "first_value": start_value,
        "last_value": end_value,
        "improvement_percentage_points": improvement * 100.0,
        "required_improvement_percentage_points": float(gate["minimum_improvement_percentage_points"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("models/yolo/experiment_configs/phase1.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="unique continuation run directory")
    args = parser.parse_args()

    from ultralytics import YOLO, __version__ as ultralytics_version

    config = yaml.safe_load(resolve(args.config).read_text())
    experiment = config["experiments"]["E4"]
    gate = continuation_gate(experiment["continuation_gate"])
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "continuation_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps({"E4_continuation_gate": gate}, indent=2), flush=True)
    if not gate["eligible"]:
        (output / "skipped.json").write_text(json.dumps({"status": "skipped", **gate}, indent=2) + "\n")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("E4 continuation requires the allocated CUDA GPU")
    source_last = resolve(experiment["detector_checkpoint"])
    if not source_last.is_file():
        raise FileNotFoundError(f"E1 resume checkpoint is missing: {source_last}")

    # Edit only a copy: E1's best.pt, last.pt, args, and result logs remain untouched.
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model = YOLO(str(source_last))
    additional_epochs = int(experiment["epochs_total"]) - int(experiment.get("source_epochs", 100))
    if additional_epochs <= 0:
        raise ValueError("E4 must request a positive number of additional detector epochs")
    model.train(
        data=str(resolve(config["dataset_yaml"])),
        epochs=additional_epochs,
        imgsz=int(config["image_size"]),
        batch=int(config["batch_size"]),
        workers=int(config["workers"]),
        device=0,
        seed=int(config["seed"]),
        deterministic=True,
        patience=additional_epochs + 1,
        project=str(output),
        name="train",
        exist_ok=False,
        pretrained=True,
        plots=True,
        val=True,
        cache=False,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    best_weights = output / "train" / "weights" / "best.pt"
    if not best_weights.is_file():
        raise RuntimeError(f"Resumed training did not produce {best_weights}")
    compute = {
        "strategy": "continue from E1 last.pt weights; initialize a fresh optimizer for additional detector epochs",
        "source_checkpoint": str(source_last),
        "best_weights": str(best_weights),
        "epochs_total": int(experiment["epochs_total"]),
        "additional_epochs": additional_epochs,
        "optimizer_state": "reinitialized because completed E1 checkpoint is stripped of optimizer state",
        "continued_from_epoch": int(gate["last_epoch"]),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_hours": elapsed / 3600.0,
        "wall_seconds": elapsed,
        "peak_vram_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "total_parameters": sum(parameter.numel() for parameter in model.model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.model.parameters() if parameter.requires_grad),
        "ultralytics_version": ultralytics_version,
        "continuation_gate": gate,
    }
    (output / "compute.json").write_text(json.dumps(compute, indent=2) + "\n")
    (output / "experiment_config.json").write_text(json.dumps(compute, indent=2) + "\n")
    print(json.dumps(compute, indent=2), flush=True)


if __name__ == "__main__":
    main()
