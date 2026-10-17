"""Cross-fit a conservative box-edge correction on clean saved predictions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sarbench.metrics import detection_metrics
from sarbench.twins import clean_image_ids, load_twins


def iou(box, boxes):
    lo = np.maximum(box[:2], boxes[:, :2])
    hi = np.minimum(box[2:], boxes[:, 2:])
    overlap = np.maximum(hi - lo, 0).prod(1)
    return overlap / np.maximum((box[2:] - box[:2]).prod() +
                                (boxes[:, 2:] - boxes[:, :2]).prod(1) - overlap, 1e-8)


def xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x+w, y+h], dtype=float)


def fit(ids, predictions, annotations, images):
    truth = defaultdict(list)
    for a in annotations:
        if a['image_id'] in ids and not a.get('iscrowd', 0):
            truth[a['image_id'], a['category_id']].append(xyxy(a['bbox']))
    proposals = defaultdict(list)
    for p in predictions:
        if p['image_id'] in ids and p['score'] >= 0.1:
            proposals[p['image_id'], p['category_id']].append(p)
    errors = defaultdict(list)
    for (image_id, category), boxes in truth.items():
        boxes = np.stack(boxes)
        used = set()
        image = images[image_id]
        scale = np.array([512/image['width'], 512/image['height']]*2)
        for p in sorted(proposals[image_id, category], key=lambda p: -p['score']):
            box = xyxy(p['bbox'])
            overlaps = iou(box, boxes)
            for index in used:
                overlaps[index] = -1
            j = int(overlaps.argmax())
            if overlaps[j] >= 0.5:
                used.add(j)
                errors[category, image['width']].append((boxes[j]-box)*scale)
    # Fixed shrinkage and clipping; no search over settings on the evaluated images.
    return {f'{category}:{width}': {'matches': len(rows),
                'edge_offsets_512': (0.5*np.clip(np.median(rows, axis=0), -2, 2)).tolist()}
            for (category, width), rows in errors.items() if len(rows) >= 12}


def apply(predictions, params, images):
    result = []
    for p in predictions:
        image = images[p['image_id']]
        row = params.get(f"{p['category_id']}:{image['width']}")
        if row is None:
            result.append(p)
            continue
        scale = np.array([image['width']/512, image['height']/512]*2)
        box = xyxy(p['bbox']) + np.array(row['edge_offsets_512'])*scale
        box[[0, 2]] = np.clip(box[[0, 2]], 0, image['width'])
        box[[1, 3]] = np.clip(box[[1, 3]], 0, image['height'])
        if box[2] > box[0] and box[3] > box[1]:
            result.append({**p, 'bbox': [float(box[0]), float(box[1]),
                                         float(box[2]-box[0]), float(box[3]-box[1])]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    gt_path = args.results / 'instances_val_holdout.json'
    gt = json.loads(gt_path.read_text())
    predictions = json.loads((args.results / 'predictions_val.json').read_text())
    images = {im['id']: im for im in gt['images']}
    clean = set(clean_image_ids(gt['images']))
    by_class = defaultdict(list)
    image_classes = {a['image_id']: a['category_id'] for a in gt['annotations']}
    twins = load_twins()
    by_name = {images[i]['file_name']: i for i in clean}
    unseen = set(clean)
    while unseen:
        group = {min(unseen)}
        pending = list(group)
        while pending:
            i = pending.pop()
            for name in twins.get(images[i]['file_name'], set()):
                j = by_name.get(name)
                if j is not None and j not in group:
                    group.add(j)
                    pending.append(j)
        unseen -= group
        assert len({image_classes[i] for i in group}) == 1
        by_class[image_classes[min(group)]].append(sorted(group))
    trials = []
    for seed in [0, 1]:
        rng = np.random.default_rng(seed)
        a, b = set(), set()
        for groups in by_class.values():
            class_a, class_b = set(), set()
            for index in rng.permutation(len(groups)):
                (class_a if len(class_a) <= len(class_b) else class_b).update(groups[index])
            a.update(class_a)
            b.update(class_b)
        assert a.isdisjoint(b) and a | b == clean
        assert not any(by_name.get(name) in b for i in a
                       for name in twins.get(images[i]['file_name'], set()))
        for train_ids, test_ids in [(a, b), (b, a)]:
            params = fit(train_ids, predictions, gt['annotations'], images)
            selected = [p for p in predictions if p['image_id'] in test_ids]
            baseline = detection_metrics(str(gt_path), selected, sorted(test_ids))
            corrected = detection_metrics(str(gt_path), apply(selected, params, images), sorted(test_ids))
            trial = {'seed': seed, 'fit_images': len(train_ids), 'evaluation_images': len(test_ids),
                     'baseline': baseline, 'corrected': corrected,
                     'mAP_gain': corrected['mAP']-baseline['mAP'], 'parameters': params}
            trials.append(trial)
            print('BOX_BIAS_TRIAL', seed, len(test_ids), 'mAP gain percentage points:',
                  100*trial['mAP_gain'], flush=True)
    accepted = all(t['mAP_gain'] >= 0.001 for t in trials)
    report = {'description': 'Half-split clean holdout cross-fitting repeated with two fixed seeds; '
              'connected twin groups stay together in each split; '
              'fixed half-median box-edge correction, clipped to 1px at 512px; >=12 matches per class/source.',
              'trials': trials, 'accepted': accepted,
              'acceptance': 'at least +0.1 mAP percentage point on every evaluation half',
              'fit_parameters': fit(clean, predictions, gt['annotations'], images) if accepted else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print('BOX_BIAS_ACCEPTED', accepted, flush=True)


if __name__ == '__main__':
    main()
