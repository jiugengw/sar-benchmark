"""Does the image classifier gain from what the detector found? Cross-fitted on the holdout.

The image classifier (a head on the frozen detector's features) never sees the detections. Yet a scene full of
confident bridge boxes is strong evidence for the class "bridge", and half of its mistakes are bridge <-> playground.
This stacks a small model on top: inputs are the classifier's 9 class probabilities plus, per class, the detector's
best box score and its counts of boxes above 0.25 and 0.5 (36 numbers per image).

It is fitted on HOLDOUT images only: on training images the detector has memorised the boxes, so its detections there
are far more confident than on new images, and a stacker fitted on them would over-trust the detector. Half of the
holdout fits it, the other half is scored, then the halves swap (twin groups stay together; three random splits).
Both the classifier alone and the stacked version are scored on exactly the same held-out halves.

    python models/yolo/classifier_stack_study.py --results results/stack-memorise-lr2/one2many
    (the folder comes from evaluate_export, which now also saves image_class_probabilities.json)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))


def detection_summary(predictions, image_ids):
    """Per image: for each of the 9 classes, best score, count >= 0.25, count >= 0.5."""
    rows = {i: np.zeros(27) for i in image_ids}
    for p in predictions:
        if p['image_id'] not in rows:
            continue
        c, s, r = p['category_id'] - 1, p['score'], rows[p['image_id']]
        r[c] = max(r[c], s)
        r[9 + c] += s >= 0.25
        r[18 + c] += s >= 0.5
    for r in rows.values():
        r[9:] = np.log1p(r[9:])
    return rows


def macro_f1(truth, predicted, classes=9):
    scores = []
    for c in range(classes):
        tp = sum(t == c and p == c for t, p in zip(truth, predicted))
        fp = sum(t != c and p == c for t, p in zip(truth, predicted))
        fn = sum(t == c and p != c for t, p in zip(truth, predicted))
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from ranking_study import twin_groups
    from sarbench.twins import clean_image_ids

    gt = json.loads((args.results / 'instances_val_holdout.json').read_text())
    ids = [im['id'] for im in gt['images']]
    clean = set(clean_image_ids(gt['images']))
    with (ROOT / 'data/course-data/train/labels.csv').open() as stream:
        truth = {int(r['image_id']): int(r['label']) for r in csv.DictReader(stream)}
    probabilities = {int(k): np.array(v) for k, v in
                     json.loads((args.results / 'image_class_probabilities.json').read_text()).items()}
    summary = detection_summary(json.loads((args.results / 'predictions_val.json').read_text()), set(ids))
    x = np.stack([np.concatenate([np.log(probabilities[i] + 1e-6), summary[i]]) for i in ids])
    y = np.array([truth[i] for i in ids])
    base = np.array([int(np.argmax(probabilities[i])) for i in ids])
    groups = twin_groups(gt['images'])
    group_of = np.array([groups[i] for i in ids])
    unique = np.array(sorted(set(group_of)))
    is_clean = np.array([i in clean for i in ids])

    variants = {'classifier alone': None,
                'stacked: probabilities + detections': list(range(36)),
                'stacked: probabilities only (control for the stacker itself)': list(range(9))}
    results = defaultdict(lambda: defaultdict(list))
    confusion_pairs = defaultdict(lambda: defaultdict(int))
    names = ('aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil_tank', 'playground', 'ship', 'wind_turbine')
    for seed in range(args.seeds):
        half = set(np.random.default_rng(seed).permutation(unique)[: len(unique) // 2])
        in_a = np.array([g in half for g in group_of])
        for name, columns in variants.items():
            predicted = base.copy()
            if columns is not None:
                for fit, apply in ((in_a, ~in_a), (~in_a, in_a)):
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
                    model.fit(x[fit][:, columns], y[fit])
                    predicted[apply] = model.predict(x[apply][:, columns])
            results[name]['full'].append(macro_f1(y, predicted))
            results[name]['clean'].append(macro_f1(y[is_clean], predicted[is_clean]))
            if seed == 0:
                for t, p in zip(y, predicted):
                    if t != p:
                        confusion_pairs[name][f'{names[t]} -> {names[p]}'] += 1
    print(f"{'variant':62s} {'F1 full':>8s} {'F1 clean':>9s}  (mean of {args.seeds} cross-fitted splits)")
    for name in variants:
        print(f"{name:62s} {100 * np.mean(results[name]['full']):8.2f} {100 * np.mean(results[name]['clean']):9.2f}")
    for name in variants:
        top = sorted(confusion_pairs[name].items(), key=lambda kv: -kv[1])[:4]
        print(f'top mistakes, {name}: ' + ', '.join(f'{k} {v}' for k, v in top))
    out = args.results.parent / 'classifier_stack_study.json'
    out.write_text(json.dumps({k: {s: v for s, v in d.items()} for k, d in results.items()}, indent=2) + '\n')


if __name__ == '__main__':
    main()
