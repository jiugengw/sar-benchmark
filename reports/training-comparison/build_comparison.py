#!/usr/bin/env python3
"""Build the offline, selectable training-run comparison report."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / "runs.json"
OUTPUT = HERE / "index.html"

PROGRESS_RE = re.compile(r"epoch\s+(\d+)\s+iter\s+(\d+)/(\d+)")
VALIDATION_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s+val:\s+accuracy\s+(?P<accuracy>[-+.\deE]+)"
    r"\s+macro_f1\s+(?P<macro_f1>[-+.\deE]+)"
    r"\s+balanced_accuracy\s+(?P<balanced_accuracy>[-+.\deE]+)"
    r"\s+mAP\s+(?P<mAP>[-+.\deE]+)\s+AP50\s+(?P<AP50>[-+.\deE]+)"
    r"\s+Δm\s+(?P<delta_m>[-+.\deE]+)"
)


def repo_path(value: str | None) -> Path | None:
    return ROOT / value if value else None


def load_metrics(entry: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    metric_path = repo_path(entry.get("metrics_path"))
    if metric_path and metric_path.is_file():
        candidates = [metric_path]
    else:
        result_root = repo_path(entry.get("results_root"))
        candidates = sorted(result_root.rglob("metrics.json")) if result_root and result_root.exists() else []
    if not candidates:
        return None, None

    path = candidates[-1]
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None
    scores = data.get("val") or data.get("final_val") or data
    keys = ("epoch", "accuracy", "macro_f1", "balanced_accuracy", "mAP", "AP50", "AP75", "delta_m")
    return {key: scores[key] for key in keys if key in scores}, str(path.relative_to(ROOT))


def read_log(entry: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    path = repo_path(entry.get("log_path"))
    if not path or not path.is_file():
        return None, None
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None, None

    progress = None
    latest_validation = None
    for line in lines:
        match = PROGRESS_RE.search(line)
        if match:
            progress = {
                "epoch": int(match.group(1)),
                "iteration": int(match.group(2)),
                "iterations": int(match.group(3)),
            }
        match = VALIDATION_RE.search(line)
        if match:
            latest_validation = {
                key: int(value) if key == "epoch" else float(value)
                for key, value in match.groupdict().items()
            }

    expected_epochs = entry.get("expected_epochs")
    if progress:
        progress["expected_epochs"] = expected_epochs
        entry["progress"] = progress
        entry["status_detail"] = (
            f"Epoch {progress['epoch']}/{expected_epochs}, "
            f"batch {progress['iteration']}/{progress['iterations']}"
        )
    if latest_validation:
        latest_validation["expected_epochs"] = expected_epochs
        return latest_validation, str(path.relative_to(ROOT))
    return None, str(path.relative_to(ROOT))


def prepare_run(entry: dict[str, Any]) -> dict[str, Any]:
    run = dict(entry)
    scores, source = load_metrics(entry)
    if scores is None and entry.get("status") == "running":
        scores, source = read_log(run)
        run["provisional"] = scores is not None
    else:
        run["provisional"] = False
        if entry.get("log_path"):
            read_log(run)

    run["metrics"] = scores
    run["metrics_source"] = source

    zip_path = repo_path(entry.get("zip_path"))
    if zip_path and zip_path.is_file():
        run["zip_href"] = Path("../../") / entry["zip_path"]
        run["zip_href"] = run["zip_href"].as_posix()
    return run


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    runs = [prepare_run(entry) for entry in manifest["runs"]]
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    data = json.dumps(runs, ensure_ascii=False).replace("</", "<\\/")
    template = (HERE / "template.html").read_text()
    page = template.replace("__TITLE__", manifest["title"])
    page = page.replace("__GENERATED_AT__", generated_at)
    page = page.replace("__RUNS__", data)
    OUTPUT.write_text(page)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(runs)} runs.")


if __name__ == "__main__":
    main()
