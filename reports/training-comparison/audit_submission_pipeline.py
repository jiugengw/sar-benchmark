#!/usr/bin/env python3
"""Score the exact submission ZIP pipeline on the labeled training holdout."""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sarbench.metrics import detection_metrics
from sarbench.twins import clean_image_ids
from models.yolo.image_head import classification_metrics


def score_same_predictions_ultralytics(ground_truth: dict, predictions: list[dict]) -> dict:
    """Isolate the metric calculator by keeping every predicted box fixed."""
    import numpy as np
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import ap_per_class, box_iou

    torch.set_num_threads(4)
    truth_by_image, predictions_by_image = defaultdict(list), defaultdict(list)
    for annotation in ground_truth["annotations"]:
        if not annotation.get("iscrowd", 0):
            truth_by_image[int(annotation["image_id"])].append(annotation)
    for prediction in predictions:
        predictions_by_image[int(prediction["image_id"])].append(prediction)
    correct, confidences, predicted_classes, true_classes = [], [], [], []
    matcher = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    for image in ground_truth["images"]:
        truth = truth_by_image[int(image["id"])]
        predicted = sorted(predictions_by_image[int(image["id"])], key=lambda row: row["score"], reverse=True)
        target_class = torch.tensor([row["category_id"] - 1 for row in truth])
        pred_class = torch.tensor([row["category_id"] - 1 for row in predicted])
        true_classes.extend(target_class.tolist())
        if not predicted:
            continue
        truth_boxes = torch.tensor([row["bbox"] for row in truth], dtype=torch.float32).reshape(-1, 4)
        pred_boxes = torch.tensor([row["bbox"] for row in predicted], dtype=torch.float32).reshape(-1, 4)
        truth_boxes[:, 2:] += truth_boxes[:, :2]
        pred_boxes[:, 2:] += pred_boxes[:, :2]
        matched = (BaseValidator.match_predictions(matcher, pred_class, target_class, box_iou(truth_boxes, pred_boxes))
                   if truth else torch.zeros((len(predicted), 10), dtype=torch.bool))
        correct.append(matched.numpy())
        confidences.extend(row["score"] for row in predicted)
        predicted_classes.extend(pred_class.tolist())
    ap = ap_per_class(np.concatenate(correct), np.asarray(confidences), np.asarray(predicted_classes),
                      np.asarray(true_classes), names={row["id"] - 1: row["name"] for row in ground_truth["categories"]})[5]
    return {"mAP": float(ap.mean()), "AP50": float(ap[:, 0].mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--official-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/course-data")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    ground_truth = json.loads(args.ground_truth.read_text())
    image_ids = [int(image["id"]) for image in ground_truth["images"]]
    names = [category["name"] for category in sorted(ground_truth["categories"], key=lambda item: item["id"])]
    with (args.data / "train/labels.csv").open(newline="") as stream:
        labels = {int(row["image_id"]): int(row["label"]) for row in csv.DictReader(stream)}
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sar-submission-holdout-") as temporary:
        temporary = Path(temporary)
        view = temporary / "data"
        (view / "val").mkdir(parents=True)
        (view / "train").mkdir()
        (view / "train/labels.csv").symlink_to((args.data / "train/labels.csv").resolve())
        for image in ground_truth["images"]:
            destination = view / "val" / image["file_name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to((args.data / "train" / image["file_name"]).resolve())
        (view / "val/images.json").write_text(json.dumps({"images": ground_truth["images"]}))
        package = temporary / "holdout.zip"
        subprocess.run([
            sys.executable, "-u", str(ROOT / "models/yolo/make_submission.py"),
            "--weights", str((args.run / "train/weights/best.pt").resolve()),
            "--classifier-checkpoint", str((args.run / "image_classifier.pt").resolve()),
            "--data", str(view), "--batch", str(args.batch), "--out", str(package),
        ], cwd=ROOT, check=True)
        with zipfile.ZipFile(package) as archive:
            predictions = json.loads(archive.read("detection.json"))
            classification = {
                int(row["image_id"]): int(row["label"])
                for row in csv.DictReader(io.StringIO(archive.read("classification.csv").decode()))
            }
        inference = json.loads(package.with_suffix(".inference.json").read_text())
    if set(classification) != set(image_ids):
        raise RuntimeError("Submission predictions do not cover the exact holdout")
    classifier = classification_metrics(
        [labels[image_id] for image_id in image_ids],
        [classification[image_id] for image_id in image_ids], names,
    )
    detection = detection_metrics(str(args.ground_truth), predictions, image_ids)
    scores = {
        **detection,
        **{key: classifier[key] for key in ("accuracy", "macro_f1", "balanced_accuracy")},
        "image_classification": classifier,
        "combined_delta_m": 50 * (classifier["macro_f1"] / 0.9331 + detection["mAP"] / 0.2613 - 2),
    }
    # The same scores on the clean holdout subset: images with no near-duplicate twin among the training images.
    clean_ids = clean_image_ids(ground_truth["images"])
    clean_classifier = classification_metrics(
        [labels[image_id] for image_id in clean_ids],
        [classification[image_id] for image_id in clean_ids], names,
    )
    clean_detection = detection_metrics(str(args.ground_truth), predictions, clean_ids)
    scores["clean_holdout"] = {
        "images": len(clean_ids),
        **clean_detection,
        **{key: clean_classifier[key] for key in ("accuracy", "macro_f1", "balanced_accuracy")},
        "image_classification": clean_classifier,
        "combined_delta_m": 50 * (clean_classifier["macro_f1"] / 0.9331 + clean_detection["mAP"] / 0.2613 - 2),
    }
    official = json.loads(args.official_scores.read_text())
    metadata = {
        "evaluation": "Exact models/terramind/make_submission.py pipeline scored on labeled holdout; COCOeval detection",
        "images": len(image_ids), "batch_size": args.batch,
        "weights_sha256": inference["weights_sha256"],
        "split": "local training holdout; disjoint from official course validation",
        "classifier_method": "jointly fine-tuned classifier; native detector box classes",
    }
    package_metadata = list(args.official_scores.parent.parent.glob("*.inference.json"))
    if len(package_metadata) == 1:
        official_inference = json.loads(package_metadata[0].read_text())
        metadata["matches_official_package_weights"] = inference["weights_sha256"] == official_inference["weights_sha256"]
        if not metadata["matches_official_package_weights"]:
            raise RuntimeError("Audit checkpoint differs from the officially scored package")
    result = {
        "report_metadata": {"configuration": {"backbone": "YOLO26m-P2 joint", "epochs": 15}},
        "val": scores, "submission_pipeline": metadata, "official_scores_percent": official,
        "local_minus_official_percentage_points": {
            key: scores[key] * 100 - official[key] for key in ("macro_f1", "mAP", "accuracy", "AP50")
        },
        "clean_minus_official_percentage_points": {
            key: scores["clean_holdout"][key] * 100 - official[key] for key in ("macro_f1", "mAP", "accuracy", "AP50")
        },
    }
    result["same_predictions_ultralytics_metrics"] = score_same_predictions_ultralytics(ground_truth, predictions)
    result["same_predictions_evaluator_delta_pp"] = {
        key: 100 * (result["same_predictions_ultralytics_metrics"][key] - scores[key])
        for key in ("mAP", "AP50")
    }
    training_classifier = json.loads((args.run / "best_classifier_metrics.json").read_text())
    result["classifier_confusion_matches_training"] = classifier["confusion_matrix"] == training_classifier["confusion_matrix"]
    shutil.copyfile(args.ground_truth, args.output / "instances_val_holdout.json")
    (args.output / "predictions_val.json").write_text(json.dumps(predictions) + "\n")
    (args.output / "image_classification_predictions.json").write_text(json.dumps(classification) + "\n")
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"val": {key: value for key, value in scores.items() if isinstance(value, (float, int))},
                      "clean_holdout": {key: value for key, value in scores["clean_holdout"].items()
                                        if isinstance(value, (float, int))},
                      "audit": metadata, "gap": result["local_minus_official_percentage_points"],
                      "clean_gap": result["clean_minus_official_percentage_points"]}, indent=2))


if __name__ == "__main__":
    main()
