#!/usr/bin/env python3
"""Build the offline, selectable training-run comparison report."""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / "runs.json"
OUTPUT = HERE / "index.html"
REVIEW_IMAGES = HERE / "images"
REVIEW_THRESHOLD = 0.3
REVIEW_IOU = 0.5
REVIEW_EXAMPLES = 48
F1_REFERENCE = 0.9331
MAP_REFERENCE = 0.2613
OFFICIAL_METRICS = ("accuracy", "macro_f1", "mAP", "AP50", "AP75", "balanced_accuracy")

ERROR_FIELDS = ("objects", "matched", "missed", "missed_small", "wrong_class", "duplicate", "localization", "background")
ERROR_WEIGHTS = {"missed_small": 3, "missed": 1, "wrong_class": 2, "duplicate": 1, "localization": 1}
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


def metric_candidates(entry: dict[str, Any]) -> list[Path]:
    metric_path = repo_path(entry.get("metrics_path"))
    if metric_path and metric_path.is_file():
        return [metric_path]
    result_root = repo_path(entry.get("results_root"))
    candidates = sorted(result_root.rglob("metrics.json")) if result_root and result_root.exists() else []
    return candidates


def load_metrics(entry: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, Path | None]:
    candidates = metric_candidates(entry)
    if not candidates:
        return None, None, None

    path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None, None
    scores = data.get("val") or data.get("final_val") or data
    keys = ("epoch", "accuracy", "macro_f1", "balanced_accuracy", "mAP", "AP50", "AP75", "AP_small",
            "AR_small", "delta_m", "combined_delta_m",
            "per_class_AP", "image_classification", "detection_derived_image_classification",
            "detection_class_policies")
    selected = {key: scores[key] for key in keys if key in scores}
    if "delta_m" not in selected:
        if "combined_delta_m" in selected:
            selected["delta_m"] = selected["combined_delta_m"]
        elif "macro_f1" in selected and "mAP" in selected:
            selected["delta_m"] = 100.0 * 0.5 * (
                (float(selected["macro_f1"]) - F1_REFERENCE) / F1_REFERENCE
                + (float(selected["mAP"]) - MAP_REFERENCE) / MAP_REFERENCE
            )
    return selected, str(path.relative_to(ROOT)), path


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


def short_configuration(config: dict[str, Any]) -> str:
    pieces = [str(config.get("backbone", "model")).capitalize(), str(config.get("adapt", ""))]
    if config.get("anchor_profile"):
        pieces.append(f"anchors={config['anchor_profile']}")
    if config.get("box_loss_weight") is not None:
        pieces.append(f"box loss={config['box_loss_weight']}")
    if config.get("photometric"):
        pieces.append("photometric")
    if config.get("small_object_sampling"):
        pieces.append("small-object sampling")
    if config.get("detail_branch"):
        pieces.append("detail branch")
    if config.get("loss_weighting"):
        pieces.append(f"{config['loss_weighting']} weights")
    if config.get("unfreeze_last"):
        pieces.append(f"unfreeze last {config['unfreeze_last']}")
    if config.get("epochs"):
        pieces.append(f"{config['epochs']} epochs")
    return "; ".join(pieces)


def discover_result_runs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if not manifest.get("discover_results", True):
        return []
    explicit_paths = {str((ROOT / entry["metrics_path"]).resolve())
                      for entry in manifest["runs"] if entry.get("metrics_path")}
    explicit_roots = [repo_path(entry.get("results_root")) for entry in manifest["runs"]]
    discovered = []
    for metric_path in sorted((ROOT / "results").rglob("metrics.json")):
        if str(metric_path.resolve()) in explicit_paths:
            continue
        if any(root and root.exists() and metric_path.is_relative_to(root) for root in explicit_roots):
            continue
        prediction_path = metric_path.with_name("predictions_val.json")
        ground_truth_path = metric_path.with_name("instances_val_holdout.json")
        if not prediction_path.is_file() or not ground_truth_path.is_file():
            continue
        try:
            data = json.loads(metric_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        config = (data.get("report_metadata", {}).get("configuration") or data.get("args") or {})
        relative = metric_path.parent.relative_to(ROOT / "results").as_posix()
        output = Path(str(config.get("out", ""))).name
        name = output if output and output not in {"runs", "results"} else relative
        if any(item["name"] == name for item in discovered):
            name = relative
        discovered.append({
            "job_id": "local-" + re.sub(r"[^A-Za-z0-9]+", "-", relative).strip("-"),
            "name": name,
            "status": "completed",
            "configuration": short_configuration(config),
            "metrics_path": metric_path.relative_to(ROOT).as_posix(),
            "predictions_path": prediction_path.relative_to(ROOT).as_posix(),
            "ground_truth_path": ground_truth_path.relative_to(ROOT).as_posix(),
            "compare_by_default": False,
        })
    return discovered


def prepare_run(entry: dict[str, Any]) -> dict[str, Any]:
    run = dict(entry)
    scores, source, metric_path = load_metrics(entry)
    if scores is None and entry.get("status") == "running":
        scores, source = read_log(run)
        run["provisional"] = scores is not None
    else:
        run["provisional"] = False
        if entry.get("log_path"):
            read_log(run)

    run["metrics"] = scores
    run["metrics_source"] = source
    official_path = repo_path(entry.get("official_scores_path"))
    if not official_path and entry.get("zip_path"):
        official_path = repo_path(entry["zip_path"]).parent / "scoring_result" / "scores.json"
    run["official_metrics"] = None
    if official_path and official_path.is_file():
        official = json.loads(official_path.read_text())
        if "delta_m" in official:
            run["official_metrics"] = {
                key: float(value) / 100 if key in OFFICIAL_METRICS else float(value)
                for key, value in official.items() if key in (*OFFICIAL_METRICS, "delta_m")
            }
            run["official_metrics_source"] = official_path.relative_to(ROOT).as_posix()
    history_path = repo_path(entry.get("training_history_path"))
    if history_path and history_path.is_file():
        with history_path.open(newline="") as stream:
            history = list(csv.DictReader(stream))
        if history:
            best = max(history, key=lambda row: float(row["metrics/combined_delta_m"]))
            run["training_metrics"] = {
                "epoch": int(best["epoch"]),
                "macro_f1": float(best["metrics/image_macro_f1"]),
                "mAP": float(best["metrics/mAP50-95(B)"]),
                "AP50": float(best["metrics/mAP50(B)"]),
                "delta_m": float(best["metrics/combined_delta_m"]),
            }
            run["training_metrics_source"] = history_path.relative_to(ROOT).as_posix()
    if entry.get("checkpoint_metrics_path"):
        checkpoint_scores, checkpoint_source, _ = load_metrics({"metrics_path": entry["checkpoint_metrics_path"]})
        run["checkpoint_metrics"] = checkpoint_scores
        run["checkpoint_metrics_source"] = checkpoint_source
    if metric_path:
        raw_metrics = json.loads(metric_path.read_text())
        run["pipeline_audit"] = {key: raw_metrics[key] for key in (
            "same_predictions_ultralytics_metrics", "same_predictions_evaluator_delta_pp",
            "classifier_confusion_matches_training", "submission_pipeline"
        ) if key in raw_metrics}
        run.setdefault("predictions_path", metric_path.with_name("predictions_val.json").relative_to(ROOT).as_posix())
        run.setdefault("ground_truth_path", metric_path.with_name("instances_val_holdout.json").relative_to(ROOT).as_posix())
    run["review_available"] = bool(
        run.get("predictions_path") and run.get("ground_truth_path")
        and repo_path(run["predictions_path"]).is_file()
        and repo_path(run["ground_truth_path"]).is_file()
    )

    zip_path = repo_path(entry.get("zip_path"))
    if zip_path and zip_path.is_file():
        run["zip_href"] = Path("../../") / entry["zip_path"]
        run["zip_href"] = run["zip_href"].as_posix()
    return run


def official_results(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep official percentages separate from fractional local holdout metrics."""
    names = {run.get("official_metrics_source"): run for run in runs if run.get("official_metrics_source")}
    results = []
    for path in sorted((ROOT / "submissions").glob("*/scoring_result/scores.json")):
        source = path.relative_to(ROOT).as_posix()
        try:
            scores = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            print(f"Skipping unreadable official scores: {source}")
            continue
        if "delta_m" not in scores:
            continue
        run = names.get(source, {})
        package = path.parent.parent
        zips = sorted(package.glob("*.zip"))
        results.append({
            "job_id": run.get("job_id", package.name),
            "name": run.get("name", package.name.replace("_", " ")),
            "metrics": {key: float(value) / 100 if key in OFFICIAL_METRICS else float(value)
                        for key, value in scores.items() if key in (*OFFICIAL_METRICS, "delta_m")},
            "source": source,
            "zip_href": "../../" + zips[0].relative_to(ROOT).as_posix() if zips else None,
        })
    return sorted(results, key=lambda result: result["metrics"]["delta_m"], reverse=True)


def box_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def diagnose(image: dict[str, Any], truth: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    used: set[int] = set()
    counts: Counter[str] = Counter({"objects": len(truth)})
    small = {index for index, ann in enumerate(truth)
             if ann["bbox"][2] * ann["bbox"][3] * 512**2 / (image["width"] * image["height"]) < 16**2}
    tagged = []
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        overlap = [box_iou(prediction["bbox"], ann["bbox"]) for ann in truth]
        same_class = [index for index, ann in enumerate(truth)
                      if ann["category_id"] == prediction["category_id"]]
        available = [index for index in same_class if index not in used and overlap[index] >= REVIEW_IOU]
        if available:
            best = max(available, key=lambda index: overlap[index])
            used.add(best)
            kind = "matched"
        elif any(overlap[index] >= REVIEW_IOU for index in same_class):
            best, kind = None, "duplicate"
        elif any(value >= REVIEW_IOU for value in overlap):
            best, kind = None, "wrong_class"
        elif any(overlap[index] >= 0.1 for index in same_class):
            best, kind = None, "localization"
        else:
            best, kind = None, "background"
        counts[kind] += 1
        if len(tagged) < 20:
            tagged.append({**prediction, "kind": kind})
    missed = set(range(len(truth))) - used
    counts["missed"] = len(missed)
    counts["missed_small"] = len(missed & small)
    counts["matched_small"] = len(small & used)
    return {"counts": dict(counts), "missed_indices": sorted(missed), "predictions": tagged}


def build_review_data(runs: list[dict[str, Any]]) -> dict[str, Any]:
    available = [run for run in runs if run["review_available"]]
    if not available:
        return {"models": [], "examples": [], "score_threshold": REVIEW_THRESHOLD, "match_iou": REVIEW_IOU}

    reference_path = repo_path(available[0]["ground_truth_path"])
    reference = json.loads(reference_path.read_text())
    reference_images = reference["images"]
    reference_annotations = reference["annotations"]
    categories = {category["id"]: category["name"] for category in reference["categories"]}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in reference_annotations:
        if not ann.get("iscrowd", 0):
            annotations[ann["image_id"]].append(ann)

    reviews_by_run: dict[str, dict[int, dict[str, Any]]] = {}
    totals_by_run: dict[str, Counter[str]] = {}
    accepted_runs = []
    for run in available:
        gt_path = repo_path(run["ground_truth_path"])
        other_gt = json.loads(gt_path.read_text())
        if other_gt.get("images") != reference_images or other_gt.get("annotations") != reference_annotations:
            print(f"Skipping visual review for {run['name']}: holdout differs.")
            run["review_available"] = False
            continue
        predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        prediction_path = repo_path(run["predictions_path"])
        for prediction in json.loads(prediction_path.read_text()):
            if prediction.get("score", 0) >= REVIEW_THRESHOLD:
                predictions_by_image[prediction["image_id"]].append(prediction)
        reviews_by_run[run["job_id"]] = {}
        totals_by_run[run["job_id"]] = Counter()
        for image in reference_images:
            image_id = image["id"]
            review = diagnose(image, annotations[image_id], predictions_by_image[image_id])
            reviews_by_run[run["job_id"]][image_id] = review
            totals_by_run[run["job_id"]].update(review["counts"])
        accepted_runs.append(run)

    if not accepted_runs:
        return {"models": [], "examples": [], "score_threshold": REVIEW_THRESHOLD, "match_iou": REVIEW_IOU}

    def severity(image: dict[str, Any]) -> float:
        truth_count = max(1, len(annotations[image["id"]]))
        per_run = []
        for run in accepted_runs:
            counts = reviews_by_run[run["job_id"]][image["id"]]["counts"]
            per_run.append(sum(weight * counts.get(key, 0) for key, weight in ERROR_WEIGHTS.items()) / truth_count)
        return max(per_run, default=0.0)

    ranked = sorted(reference_images, key=severity, reverse=True)
    selected: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    for image in ranked:
        truth = annotations[image["id"]]
        category = categories[truth[0]["category_id"]] if truth else "unlabeled"
        if class_counts[category] < 3 and len(selected) < REVIEW_EXAMPLES:
            selected.append(image)
            class_counts[category] += 1
    selected_ids = {image["id"] for image in selected}
    selected.extend(image for image in ranked if image["id"] not in selected_ids)
    selected = selected[:REVIEW_EXAMPLES]

    REVIEW_IMAGES.mkdir(parents=True, exist_ok=True)
    examples = []
    images_root = ROOT / "data" / "course-data" / "train"
    for image in selected:
        image_id = image["id"]
        truth = annotations[image_id]
        category = categories[truth[0]["category_id"]] if truth else "unlabeled"
        source = images_root / image["file_name"]
        destination = REVIEW_IMAGES / f"{image_id}{source.suffix.lower()}"
        shutil.copyfile(source, destination)
        examples.append({
            "id": image_id,
            "file_name": image["file_name"],
            "category": category,
            "width": image["width"],
            "height": image["height"],
            "image_href": f"images/{image_id}{source.suffix.lower()}",
            "truth": [{"bbox": ann["bbox"], "category_id": ann["category_id"]} for ann in truth],
            "reviews": {run["job_id"]: reviews_by_run[run["job_id"]][image_id] for run in accepted_runs},
        })

    model_data = []
    for run in accepted_runs:
        counts = totals_by_run[run["job_id"]]
        model_data.append({
            "job_id": run["job_id"],
            "name": run["name"],
            "total": {key: counts.get(key, 0) for key in ERROR_FIELDS},
            "per_class_AP": (run.get("metrics") or {}).get("per_class_AP", {}),
        })
    return {"models": model_data, "examples": examples, "score_threshold": REVIEW_THRESHOLD,
            "match_iou": REVIEW_IOU, "holdout_images": len(reference_images), "categories": list(categories.values())}


def main() -> None:
    global REVIEW_THRESHOLD, REVIEW_IOU, REVIEW_EXAMPLES
    manifest = json.loads(MANIFEST.read_text())
    review_config = manifest.get("visual_review", {})
    REVIEW_THRESHOLD = float(review_config.get("score_threshold", REVIEW_THRESHOLD))
    REVIEW_IOU = float(review_config.get("match_iou", REVIEW_IOU))
    REVIEW_EXAMPLES = int(review_config.get("examples", REVIEW_EXAMPLES))
    entries = manifest["runs"] + discover_result_runs(manifest)
    runs = [prepare_run(entry) for entry in entries]
    official = official_results(runs)
    review_data = build_review_data(runs)
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    data = json.dumps(runs, ensure_ascii=False).replace("</", "<\\/")
    review_json = json.dumps(review_data, ensure_ascii=False).replace("</", "<\\/")
    template = (HERE / "template.html").read_text()
    page = template.replace("__TITLE__", manifest["title"])
    page = page.replace("__GENERATED_AT__", generated_at)
    page = page.replace("__RUNS__", data)
    page = page.replace("__REVIEW_DATA__", review_json)
    page = page.replace("__OFFICIAL_RESULTS__", json.dumps(official, ensure_ascii=False).replace("</", "<\\/"))
    target_config = {"delta_m": manifest.get("target_delta_m", 30),
                     "f1_reference": F1_REFERENCE, "map_reference": MAP_REFERENCE,
                     "job_id": manifest.get("target_run_job_id")}
    page = page.replace("__TARGET_CONFIG__", json.dumps(target_config))
    OUTPUT.write_text(page)
    audit = {
        "generated_at": generated_at,
        "target": target_config,
        "runs": [{key: run[key] for key in (
            "job_id", "name", "metrics", "metrics_source", "official_metrics", "official_metrics_source",
            "training_metrics", "training_metrics_source", "checkpoint_metrics", "checkpoint_metrics_source",
            "zip_href", "predictions_path", "ground_truth_path", "pipeline_audit"
        ) if key in run} for run in runs if run.get("official_metrics")],
    }
    (HERE / "score-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(runs)} runs and {len(review_data['examples'])} visual examples.")


if __name__ == "__main__":
    main()
