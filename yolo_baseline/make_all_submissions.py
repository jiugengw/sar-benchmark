#!/usr/bin/env python3
"""Package every completed YOLO run as a separate project-format prediction ZIP."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("yolo11s", "yolo11m", "yolo11n", "yolo11l", "yolo26s", "yolo26m", "yolo26l")
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submissions" / "161026" / "tests-no-submit"


def completed_run(model: str) -> tuple[Path, Path, str]:
    candidates = sorted(RESULTS.glob(f"{model}-baseline-*/seed_0/metrics.json"))
    valid: list[tuple[float, Path, Path, str]] = []
    for metrics_path in candidates:
        try:
            data = json.loads(metrics_path.read_text())
            config = data.get("report_metadata", {}).get("configuration", {})
            training = data.get("training", {})
            epochs = int(training.get("completed_epochs", 0))
            weights = Path(training.get("best_weights", ""))
            if not weights.is_absolute():
                weights = ROOT / weights
            if epochs >= 24 and weights.is_file() and config.get("backbone", "").lower() == model:
                run_name = metrics_path.parents[1].name
                valid.append((metrics_path.stat().st_mtime, metrics_path, weights, run_name))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    if not valid:
        raise FileNotFoundError(f"No completed 24-epoch result and checkpoint found for {model}")
    _mtime, metrics_path, weights_path, run_name = max(valid, key=lambda item: item[0])
    return metrics_path, weights_path, run_name


def main() -> None:
    for model in MODELS:
        metrics_path, weights, run_name = completed_run(model)
        output_dir = SUBMISSIONS / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "scoring_result").mkdir(exist_ok=True)
        (output_dir / "scoring_result" / ".gitkeep").touch(exist_ok=True)
        output_zip = output_dir / f"{model}_baseline_{run_name.rsplit('-', 1)[-1]}.zip"
        if output_zip.is_file():
            print(f"Already packaged {model}: {output_zip.relative_to(ROOT)}", flush=True)
            continue
        print(f"Packaging {model} from {metrics_path.relative_to(ROOT)}", flush=True)
        subprocess.run([
            sys.executable,
            str(ROOT / "yolo_baseline" / "make_submission.py"),
            "--weights", str(weights),
            "--data", str(ROOT / "data" / "course-data"),
            "--classes", str(ROOT / "data" / "yolo11s-sar-512" / "conversion_manifest.json"),
            "--out", str(output_zip),
            "--batch", "16",
            "--workers", "4",
        ], cwd=ROOT, check=True)
    print("All YOLO model ZIPs are ready; none have been submitted.", flush=True)


if __name__ == "__main__":
    main()
