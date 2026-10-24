"""Which boxes, in which order, should fill the bottom of the ranking? (builds on hedge_study.py)

Three kinds of tail boxes, all ranked below every submitted box (scores >= 0.001), each kind in its own band so that
the order of the kinds is exact:
  suppressed  the detector's own candidates that duplicate removal (NMS) dropped: learned alternative boxes
              (raw_candidates.py), the Soft-NMS idea of demoting instead of deleting
  sizes       each submitted box 8% larger and 8% smaller (hedge_study.py)
  shifts      each submitted box shifted by 8% of its size left, right, up and down
Every submitted box is kept; tail boxes only fill an image and class up to 100 boxes, the scorer's limit. Scored as
official-mix mAP on two holdout halves (twin groups together), against the submitted boxes alone.

    python models/yolo/tail_study.py results/ft40-m4rep-bp4-memorise-335138/one2many
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
sys.path.insert(0, str(ROOT / 'models/yolo'))


def iou_one(a, b):
    w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return w * h / (a[2] * a[3] + b[2] * b[3] - w * h + 1e-9)


def suppressed_boxes(submitted: list[dict], raw: list[dict], min_iou: float = 0.0) -> list[dict]:
    """Raw candidates that are not among the submitted boxes; with min_iou, only those overlapping a submitted box of
    the same class at least that much (alternatives to a found object rather than separate guesses)."""
    seen = {(p['image_id'], p['category_id'], round(p['score'], 5)) for p in submitted}
    kept = defaultdict(list)
    for p in submitted:
        kept[(p['image_id'], p['category_id'])].append(p['bbox'])
    out = []
    for p in raw:
        if (p['image_id'], p['category_id'], round(p['score'], 5)) in seen:
            continue
        if min_iou and max((iou_one(p['bbox'], b) for b in kept[(p['image_id'], p['category_id'])]),
                           default=0.0) < min_iou:
            continue
        out.append(p)
    return out


def stack(submitted: list[dict], tiers: list[list[dict]], max_per_class: int = 100) -> list[dict]:
    """Submitted boxes, then each tier in its own score band below the previous one (order within a tier kept)."""
    room = defaultdict(lambda: max_per_class)
    for p in submitted:
        room[(p['image_id'], p['category_id'])] -= 1
    out = list(submitted)
    for k, tier in enumerate(tiers):
        band = 0.0009 * 0.001 ** k
        for p in sorted(tier, key=lambda p: -p['score']):
            key = (p['image_id'], p['category_id'])
            if room[key] > 0:
                room[key] -= 1
                out.append({**p, 'score': band * max(min(p['score'], 1.0), 1e-6)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    from hedge_study import SHIFTS, SIZES, hedge
    from official_mix_score import NAMES, official_shares
    from ranking_study import twin_groups
    from resolution_probe import coco
    from sarbench.twins import clean_image_ids

    gt_path = args.results / 'instances_val_holdout.json'
    gt = json.loads(gt_path.read_text())
    images = {im['id']: im for im in gt['images']}
    clean = set(clean_image_ids(gt['images']))
    submitted = json.loads((args.results / 'predictions_val.json').read_text())
    raw = json.loads((args.results / 'raw_candidates.json').read_text())
    shares = official_shares(ROOT / 'submissions/memorise-lr2-332326/memorise-lr2-332326.zip',
                             ROOT / 'data/course-data/val/images.json')
    groups = twin_groups(gt['images'])
    unique = sorted({groups[i] for i in clean})
    half = set(np.random.default_rng(0).permutation(unique)[: len(unique) // 2])
    halves = [{i for i in clean if groups[i] in half}, {i for i in clean if groups[i] not in half}]

    def mix(boxes, ids):
        small = {i for i in ids if images[i]['width'] == 256}
        with contextlib.redirect_stdout(io.StringIO()):
            _, ap_s = coco(gt_path, boxes, small)
            _, ap_l = coco(gt_path, boxes, ids - small)
        values = []
        for n in NAMES:
            s, l = ap_s.get(n, float('nan')), ap_l.get(n, float('nan'))
            values.append(l if s != s else s if l != l else shares[n] * s + (1 - shares[n]) * l)
        return 100 * float(np.mean(values))

    copies = lambda transforms: [p for p in hedge(submitted, transforms, 0.0, 1e-9, max_per_class=10 ** 9)
                                 if p['score'] < 1e-6]
    sizes, shifts = copies(SIZES), copies(SHIFTS)
    supp = suppressed_boxes(submitted, raw)
    near = suppressed_boxes(submitted, raw, min_iou=0.3)
    print(f'{len(submitted)} submitted boxes, {len(raw)} raw candidates, {len(supp)} suppressed '
          f'({len(near)} overlapping a submitted box by IoU >= 0.3)', flush=True)
    base = [mix(submitted, h) for h in halves]
    print(f'submitted only: official-mix {base[0]:.2f} / {base[1]:.2f}', flush=True)
    variants = {
        'sizes, shifts (current)': [sizes, shifts],
        'suppressed': [supp],
        'suppressed near': [near],
        'suppressed, sizes, shifts': [supp, sizes, shifts],
        'near, sizes, shifts': [near, sizes, shifts],
        'sizes, near, shifts': [sizes, near, shifts],
        'sizes, shifts, near': [sizes, shifts, near],
        'sizes, shifts, suppressed': [sizes, shifts, supp],
    }
    for name, tiers in variants.items():
        boxes = stack(submitted, tiers)
        gains = [mix(boxes, h) - b for h, b in zip(halves, base)]
        print(f'  {name:28s} gain {gains[0]:+.2f} / {gains[1]:+.2f}  (mean {np.mean(gains):+.2f})', flush=True)


if __name__ == '__main__':
    main()
