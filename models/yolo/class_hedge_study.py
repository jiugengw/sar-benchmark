"""Class hedging: copies of each box relabelled as the image's other likely classes, ranked at the bottom.

mAP is computed per class, and the scorer keeps up to 100 boxes per image AND class. A box relabelled as class c
therefore uses class c's budget, not its own class's, and ranked below every real class-c box it cannot lower class
c's AP (AP takes the best precision at each recall level; boxes appended below all others only add points to the
right of the curve). Every image holds one class and our mistakes are mostly bridge <-> playground <-> airport, so a
box the detector found but called the wrong class can still count for the right one.

Copies are ranked by the image classifier's probability for the new class times the box score, below the box-hedge
tiers or between them and the originals (both are tried). Measured on the holdout as official-mix mAP, cross-fitted
over two halves like hedge_study.py.

    python models/yolo/class_hedge_study.py results/ft40-m4rep-bp4-memorise-335138/one2many
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


def class_hedge(predictions: list[dict], probabilities: dict, top: float, min_probability: float = 0.0,
                max_per_class: int = 100, top_classes: int = 8) -> list[dict]:
    """For every box (class k, score s) in image i, a copy for each other class c with p_i(c) >= min_probability (the
    top_classes most likely ones), scored top * s * p_i(c). Copies only fill each image and class up to max_per_class."""
    room = defaultdict(lambda: max_per_class)
    for p in predictions:
        room[(p['image_id'], p['category_id'])] -= 1
    copies = []
    for p in predictions:
        probs = probabilities.get(str(p['image_id']))
        if probs is None:
            continue
        order = np.argsort(probs)[::-1]
        for c in [int(c) + 1 for c in order[:top_classes]]:
            if c == p['category_id'] or probs[c - 1] < min_probability:
                continue
            copies.append({**p, 'category_id': c, 'score': top * p['score'] * probs[c - 1]})
    out = list(predictions)
    for p in sorted(copies, key=lambda p: -p['score']):
        key = (p['image_id'], p['category_id'])
        if room[key] > 0:
            room[key] -= 1
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    from hedge_study import CHOSEN_TIERS, hedge_tiers
    from official_mix_score import NAMES, official_shares
    from ranking_study import twin_groups
    from resolution_probe import coco
    from sarbench.twins import clean_image_ids

    gt_path = args.results / 'instances_val_holdout.json'
    gt = json.loads(gt_path.read_text())
    images = {im['id']: im for im in gt['images']}
    clean = set(clean_image_ids(gt['images']))
    predictions = json.loads((args.results / 'predictions_val.json').read_text())
    probabilities = json.loads((args.results / 'image_class_probabilities.json').read_text())
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

    boxes_hedged = hedge_tiers(predictions, CHOSEN_TIERS)
    base = [mix(predictions, h) for h in halves]
    ref = [mix(boxes_hedged, h) for h in halves]
    print(f'no hedging {base[0]:.2f} / {base[1]:.2f}; box hedging {ref[0]:.2f} / {ref[1]:.2f}', flush=True)
    variants = {
        # class copies below the originals and all box-hedge copies (1e-12 << every box-hedge score)
        'class copies last': class_hedge(boxes_hedged, probabilities, 1e-12),
        'class copies last, p >= 0.01': class_hedge(boxes_hedged, probabilities, 1e-12, 0.01),
        'class copies last, top 2 classes': class_hedge(boxes_hedged, probabilities, 1e-12, 0.0, top_classes=2),
        # class copies directly below the originals, above the box-hedge copies
        'class copies first': hedge_tiers(class_hedge(predictions, probabilities, 0.0009 * 0.999), CHOSEN_TIERS),
    }
    for name, boxes in variants.items():
        gains = [mix(boxes, h) - r for h, r in zip(halves, ref)]
        print(f'  {name:34s} gain over box hedging {gains[0]:+.2f} / {gains[1]:+.2f}', flush=True)


if __name__ == '__main__':
    main()
