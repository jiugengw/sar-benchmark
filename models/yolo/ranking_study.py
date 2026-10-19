"""Rank boxes by how likely they are to count, not by raw confidence. Clean holdout, cross-fitted, no GPU.

COCO AP sorts each class's boxes from all images by score and rewards correct boxes ranked early, at ten IoU
thresholds 0.50-0.95. The detector's confidence is one number per box; whether a box clears the strict thresholds
also depends on things the confidence ignores, which our audits measured:
- the image source (256 px vs 512 px: e.g. oil tank 63.6 vs 21.6 AP),
- the box size (a 1 px error ruins an 8 px box, not a 40 px one),
- whether a higher-scored box of the same class already covers the same object (a duplicate counts as a false alarm).
A narrower variant keeps each (class, source) cell in the detector's order (isotonic fit of hit rate on score) and only
re-interleaves the sources.

For every predicted box, the target is its hit rate: the share of the ten thresholds at which COCO's greedy matching
makes it a true positive. A small gradient-boosted model learns that hit rate from (score, class, source, size, shape,
overlap with higher-scored boxes, rank in its image) on one half of the clean holdout; the other half is rescored with
the prediction and evaluated, then the halves swap (twin groups stay together; three random splits). The rescored
boxes are the same boxes; only their order changes. One model, one pass, deterministic post-processing.

    python models/yolo/ranking_study.py --results results/memorise-lr2-332326/one2many
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

THRESHOLDS = np.linspace(0.5, 0.95, 10)


def xyxy(b):
    return np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], dtype=float)


def iou_matrix(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def hit_rates(predictions, gt_boxes):
    """Share of the ten thresholds at which each prediction is a COCO true positive (greedy, by score, per image/class)."""
    hits = np.zeros(len(predictions))
    groups = defaultdict(list)
    for k, p in enumerate(predictions):
        groups[(p['image_id'], p['category_id'])].append(k)
    for key, members in groups.items():
        gts = gt_boxes.get(key)
        if not gts:
            continue
        members = sorted(members, key=lambda k: -predictions[k]['score'])
        ious = iou_matrix(np.stack([xyxy(predictions[k]['bbox']) for k in members]), np.stack(gts))
        for t in THRESHOLDS:
            taken = np.zeros(len(gts), bool)
            for row, k in enumerate(members):
                candidates = np.where(~taken & (ious[row] >= t))[0]
                if len(candidates):
                    taken[candidates[np.argmax(ious[row, candidates])]] = True
                    hits[k] += 1 / len(THRESHOLDS)
    return hits


def features(predictions, widths):
    """score, logit, class, source, log size, log aspect, overlap with a higher-scored same-class box, rank."""
    rows = np.zeros((len(predictions), 8))
    groups = defaultdict(list)
    for k, p in enumerate(predictions):
        groups[(p['image_id'], p['category_id'])].append(k)
    for members in groups.values():
        members = sorted(members, key=lambda k: -predictions[k]['score'])
        boxes = np.stack([xyxy(predictions[k]['bbox']) for k in members])
        ious = iou_matrix(boxes, boxes)
        for rank, (row, k) in enumerate(zip(range(len(members)), members)):
            p = predictions[k]
            w, h = max(p['bbox'][2], 1e-3), max(p['bbox'][3], 1e-3)
            s = min(max(p['score'], 1e-6), 1 - 1e-6)
            width = widths[p['image_id']]
            rows[k] = [s, np.log(s / (1 - s)), p['category_id'], 0 if width == 512 else 1 if width == 256 else 2,
                       0.5 * np.log(w * h), np.log(w / h), ious[row, :row].max() if row else 0.0, rank]
    return rows


def twin_groups(images):
    """Union of twin pairs among these images: images in one group stay in the same half."""
    from sarbench.twins import load_twins

    twins = load_twins()
    name_to_id = {im['file_name']: im['id'] for im in images}
    parent = {im['id']: im['id'] for im in images}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for name, others in twins.items():
        for other in others:
            if name in name_to_id and other in name_to_id:
                parent[find(name_to_id[name])] = find(name_to_id[other])
    return {i: find(i) for i in parent}


def coco_map(gt_path, predictions, image_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        ev = COCOeval(gt, gt.loadRes([dict(p) for p in predictions]), 'bbox')
        ev.params.imgIds = sorted(image_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    precision = ev.eval['precision'][:, :, :, 0, -1]
    names = {c['id']: c['name'] for c in gt.dataset['categories']}
    return ev.stats[0], {names[c]: float(precision[:, :, k].mean()) for k, c in enumerate(ev.params.catIds)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=ROOT / 'results/memorise-lr2-332326/one2many',
                        help='predictions_val.json and instances_val_holdout.json of a holdout evaluation')
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.isotonic import IsotonicRegression
    from sarbench.twins import clean_image_ids

    gt_path = args.results / 'instances_val_holdout.json'
    gt = json.loads(gt_path.read_text())
    clean = set(clean_image_ids(gt['images']))
    widths = {im['id']: im['width'] for im in gt['images']}
    predictions = [p for p in json.loads((args.results / 'predictions_val.json').read_text()) if p['image_id'] in clean]
    gt_boxes = defaultdict(list)
    for a in gt['annotations']:
        if a['image_id'] in clean:
            gt_boxes[(a['image_id'], a['category_id'])].append(xyxy(a['bbox']))
    target = hit_rates(predictions, gt_boxes)
    x = features(predictions, widths)
    groups = twin_groups([im for im in gt['images'] if im['id'] in clean])
    group_of = np.array([groups[p['image_id']] for p in predictions])
    unique_groups = np.array(sorted(set(group_of)))

    base_map, base_per_class = coco_map(gt_path, predictions, clean)
    print(f'{len(predictions)} clean-holdout boxes, {len(clean)} images, {len(unique_groups)} twin groups; '
          f'mean hit rate {target.mean():.3f}')
    print(f'baseline (detector confidence)        clean mAP {100 * base_map:.2f}', flush=True)
    variants = {'hit rate': lambda h, s: h, 'sqrt(hit rate x score)': lambda h, s: np.sqrt(h * s),
                'per class+source monotone': lambda h, s: h}
    results = {name: [] for name in variants}
    per_class = {name: defaultdict(list) for name in variants}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        half = set(rng.permutation(unique_groups)[: len(unique_groups) // 2])
        in_a = np.array([g in half for g in group_of])
        predicted = np.zeros(len(predictions))
        monotone = np.zeros(len(predictions))
        for fit, apply in ((in_a, ~in_a), (~in_a, in_a)):
            model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                  min_samples_leaf=40, categorical_features=[2, 3], random_state=0)
            model.fit(x[fit], target[fit])
            predicted[apply] = np.clip(model.predict(x[apply]), 0, 1)
            for key in set(map(tuple, x[:, 2:4])):  # per (class, source): hit rate as a monotone function of score
                cell = (x[:, 2] == key[0]) & (x[:, 3] == key[1])
                if (fit & cell).sum() < 30:
                    monotone[apply & cell] = x[apply & cell, 0]
                    continue
                iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
                iso.fit(x[fit & cell, 0], target[fit & cell])
                monotone[apply & cell] = iso.predict(x[apply & cell, 0])
        for name, combine in variants.items():
            source = monotone if name.startswith('per class') else predicted
            new = combine(source, x[:, 0]) + 1e-6 * x[:, 0]  # original score breaks ties
            rescored = [{**p, 'score': float(v)} for p, v in zip(predictions, new)]
            m, pc = coco_map(gt_path, rescored, clean)
            results[name].append(m)
            for k, v in pc.items():
                per_class[name][k].append(v)
            print(f'seed {seed}: {name:28s} clean mAP {100 * m:.2f} ({100 * (m - base_map):+.2f})', flush=True)
    print()
    for name in variants:
        print(f'{name:28s} mean clean mAP {100 * np.mean(results[name]):.2f} '
              f'({100 * (np.mean(results[name]) - base_map):+.2f}), per class change: '
              + ', '.join(f'{k} {100 * (np.mean(v) - base_per_class[k]):+.1f}' for k, v in per_class[name].items()))
    out = args.results.parent / 'ranking_study.json'
    out.write_text(json.dumps({'baseline': base_map, 'variants': {k: v for k, v in results.items()}}, indent=2) + '\n')


if __name__ == '__main__':
    main()
