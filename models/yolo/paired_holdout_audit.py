"""Paired, image-stratified uncertainty checks for saved holdout predictions.

This describes sampling uncertainty on the existing holdout. It cannot undo
earlier model selection on that holdout or guarantee an official leaderboard score.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np


def load_image_predictions(folder, image_ids):
    values = json.loads((folder / 'image_classification_predictions.json').read_text())
    labels = []
    for image_id in image_ids:
        value = values[str(image_id)]
        labels.append(int(value['label'] if isinstance(value, dict) else value))
    return np.asarray(labels)


class CachedCOCO:
    """Cache COCO's exact per-image matching, then resample those image records."""

    def __init__(self, ground_truth_path, prediction_path, image_ids):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        with contextlib.redirect_stdout(io.StringIO()):
            truth = COCO(str(ground_truth_path))
            detections = truth.loadRes(json.loads(prediction_path.read_text()))
            evaluator = COCOeval(truth, detections, 'bbox')
            evaluator.params.imgIds = image_ids
            evaluator.params.maxDets = [100]
            evaluator.params.areaRng = [[0, 1e10]]
            evaluator.params.areaRngLbl = ['all']
            evaluator.evaluate()
            evaluator.accumulate()
        precision = evaluator.eval['precision']
        self.official_ap = float(precision[precision > -1].mean())
        self.categories = evaluator.params.catIds
        self.records = []
        count = len(image_ids)
        for category_index, _category in enumerate(self.categories):
            category = []
            for image_index in range(count):
                item = evaluator.evalImgs[category_index * count + image_index]
                if item is None:
                    category.append(None)
                else:
                    category.append((np.asarray(item['dtScores']), item['dtMatches'] > 0,
                                     np.asarray(item['dtIgnore'], dtype=bool),
                                     int(np.count_nonzero(np.asarray(item['gtIgnore']) == 0))))
            self.records.append(category)
        actual = self.score(np.arange(count))
        if abs(actual - self.official_ap) > 1e-10:
            raise RuntimeError(f'Bootstrap accumulation differs from COCO: {actual} vs {self.official_ap}')

    def score(self, indices):
        aps = []
        for category in self.records:
            selected = [category[int(index)] for index in indices if category[int(index)] is not None]
            positives = sum(item[3] for item in selected)
            if not positives:
                continue
            scores = np.concatenate([item[0] for item in selected])
            if not len(scores):
                aps.append(0.0)
                continue
            matches = np.concatenate([item[1] for item in selected], axis=1)
            ignored = np.concatenate([item[2] for item in selected], axis=1)
            order = np.argsort(-scores, kind='mergesort')
            tp = np.cumsum((matches & ~ignored)[:, order], axis=1, dtype=float)
            fp = np.cumsum((~matches & ~ignored)[:, order], axis=1, dtype=float)
            recall = tp / positives
            precision = tp / (tp + fp + np.spacing(1))
            precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
            sampled = np.zeros((10, 101))
            for threshold in range(10):
                positions = np.searchsorted(recall[threshold], np.linspace(0, 1, 101), side='left')
                valid = positions < len(scores)
                sampled[threshold, valid] = precision[threshold, positions[valid]]
            aps.append(float(sampled.mean()))
        return float(np.mean(aps))


def macro_f1(truth, predictions, indices):
    matrix = np.bincount((truth[indices] * 9 + predictions[indices]), minlength=81).reshape(9, 9)
    return float(np.mean(2 * np.diag(matrix) / (matrix.sum(0) + matrix.sum(1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=200)
    args = parser.parse_args()
    truth_path = args.baseline / 'instances_val_holdout.json'
    truth = json.loads(truth_path.read_text())
    candidate_truth = json.loads((args.candidate / 'instances_val_holdout.json').read_text())
    if truth != candidate_truth:
        raise RuntimeError('Candidate and baseline do not use exactly the same holdout annotations')
    image_ids = sorted(int(image['id']) for image in truth['images'])
    with Path('data/course-data/train/labels.csv').open(newline='') as stream:
        labels = {int(row['image_id']): int(row['label']) for row in csv.DictReader(stream)}
    classes = np.asarray([labels[image_id] for image_id in image_ids])
    reference_labels = load_image_predictions(args.baseline, image_ids)
    candidate_labels = load_image_predictions(args.candidate, image_ids)
    reference = CachedCOCO(truth_path, args.baseline / 'predictions_val.json', image_ids)
    candidate = CachedCOCO(truth_path, args.candidate / 'predictions_val.json', image_ids)
    all_indices = np.arange(len(image_ids))
    point = {
        'baseline_mAP': reference.official_ap, 'candidate_mAP': candidate.official_ap,
        'baseline_macro_f1': macro_f1(classes, reference_labels, all_indices),
        'candidate_macro_f1': macro_f1(classes, candidate_labels, all_indices),
    }
    rng = np.random.default_rng(20261017)
    strata = [np.flatnonzero(classes == category) for category in range(9)]
    gains = []
    for iteration in range(args.replicates):
        indices = np.sort(np.concatenate([rng.choice(group, len(group), replace=True) for group in strata]))
        map_gain = candidate.score(indices) - reference.score(indices)
        f1_gain = macro_f1(classes, candidate_labels, indices) - macro_f1(classes, reference_labels, indices)
        gains.append((map_gain, f1_gain, 50 * (f1_gain / 0.9331 + map_gain / 0.2613)))
        if (iteration + 1) % 25 == 0:
            print(f'Paired bootstrap {iteration + 1}/{args.replicates}', flush=True)
    gains = np.asarray(gains)
    intervals = {}
    for index, name in enumerate(['mAP_gain', 'macro_f1_gain', 'delta_m_gain']):
        low, high = np.quantile(gains[:, index], [0.025, 0.975])
        intervals[name] = {'lower_95': float(low), 'upper_95': float(high),
                           'bootstrap_mean': float(gains[:, index].mean())}
    report = {
        'method': 'Paired bootstrap of images, stratified by image class; COCO AP matching cached per image',
        'images': len(image_ids), 'replicates': args.replicates, 'random_seed': 20261017,
        'point_estimates': point, 'gain_intervals': intervals,
        'detection_gain_lower_bound_positive': intervals['mAP_gain']['lower_95'] > 0,
        'limitation': 'Conditional on the existing holdout and selected models. Does not correct earlier adaptive selection or estimate official split shift.',
        'coco_accumulation_equivalence_checked': True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['mAP_gain', 'macro_f1_gain', 'delta_m_gain'])
        writer.writerows(gains.tolist())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
