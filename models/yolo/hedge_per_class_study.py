"""Per-class box hedging: pick each class's copy recipe on one holdout half, score it on the other.

A class's AP depends only on its own boxes and copies, so recipes can be chosen per class independently. Recipes are
tiers of copies (see hedge_study.py), here also with copies in whole pixels (+-1 px), which may suit 8 px wind
turbines better than 8% of their size. Reported: per-class choice and the cross-fitted official-mix gain over the
current shared recipe (sizes then shifts, 8%).

    python models/yolo/hedge_per_class_study.py results/ft40-m4rep-bp4-memorise-335138/one2many
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


def transform(box, t):
    """t = (sw, sh, dx, dy) relative to the box size, or ('px', dw, dh, dx, dy) in pixels."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    if t[0] == 'px':
        _, dw, dh, dx, dy = t
        nw, nh = max(w + dw, 0.5), max(h + dh, 0.5)
    else:
        sw, sh, dx, dy = (t + (0.0, 0.0))[:4] if len(t) == 2 else t
        nw, nh, dx, dy = w * sw, h * sh, dx * w, dy * h
    return [cx + dx - nw / 2, cy + dy - nh / 2, nw, nh]


def hedge_recipe(predictions: list[dict], tiers: list[list[tuple]], max_per_class: int = 100) -> list[dict]:
    """Every original kept; tier k copies ranked below tier k-1 and all originals; fill up to max_per_class."""
    room = defaultdict(lambda: max_per_class)
    for p in predictions:
        room[(p['image_id'], p['category_id'])] -= 1
    out = list(predictions)
    ordered = sorted(predictions, key=lambda p: -p['score'])
    for k, tier in enumerate(tiers):
        band = 0.0009 * 0.001 ** k
        for p in ordered:
            key = (p['image_id'], p['category_id'])
            for t in tier:
                if room[key] <= 0:
                    break
                room[key] -= 1
                out.append({**p, 'bbox': transform(p['bbox'], t), 'score': band * p['score']})
    return out


S8 = [(1.08, 1.08), (0.92, 0.92)]
SH8 = [(1, 1, 0.08, 0), (1, 1, -0.08, 0), (1, 1, 0, 0.08), (1, 1, 0, -0.08)]
RECIPES = {
    'current: sizes 8%, shifts 8%': [S8, SH8],
    'sizes 5%, shifts 5%': [[(1.05, 1.05), (0.95, 0.95)], [(1, 1, 0.05, 0), (1, 1, -0.05, 0), (1, 1, 0, 0.05), (1, 1, 0, -0.05)]],
    'sizes 12%, shifts 12%': [[(1.12, 1.12), (0.88, 0.88)], [(1, 1, 0.12, 0), (1, 1, -0.12, 0), (1, 1, 0, 0.12), (1, 1, 0, -0.12)]],
    'shifts 8%, sizes 8%': [SH8, S8],
    'sizes 8%, shifts 8%, w/h 8%': [S8, SH8, [(1.08, 0.92), (0.92, 1.08)]],
    'px: sizes +-1, shifts +-1': [[('px', 1, 1, 0, 0), ('px', -1, -1, 0, 0)],
                                  [('px', 0, 0, 1, 0), ('px', 0, 0, -1, 0), ('px', 0, 0, 0, 1), ('px', 0, 0, 0, -1)]],
    'px: sizes +-2, shifts +-1': [[('px', 2, 2, 0, 0), ('px', -2, -2, 0, 0)],
                                  [('px', 0, 0, 1, 0), ('px', 0, 0, -1, 0), ('px', 0, 0, 0, 1), ('px', 0, 0, 0, -1)]],
    'none': [],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    from official_mix_score import NAMES, official_shares
    from ranking_study import twin_groups
    from resolution_probe import coco
    from sarbench.twins import clean_image_ids

    gt_path = args.results / 'instances_val_holdout.json'
    gt = json.loads(gt_path.read_text())
    images = {im['id']: im for im in gt['images']}
    clean = set(clean_image_ids(gt['images']))
    predictions = json.loads((args.results / 'predictions_val.json').read_text())
    shares = official_shares(ROOT / 'submissions/memorise-lr2-332326/memorise-lr2-332326.zip',
                             ROOT / 'data/course-data/val/images.json')
    groups = twin_groups(gt['images'])
    unique = sorted({groups[i] for i in clean})
    half = set(np.random.default_rng(0).permutation(unique)[: len(unique) // 2])
    halves = [{i for i in clean if groups[i] in half}, {i for i in clean if groups[i] not in half}]

    def per_class_mix(boxes, ids):
        small = {i for i in ids if images[i]['width'] == 256}
        with contextlib.redirect_stdout(io.StringIO()):
            _, ap_s = coco(gt_path, boxes, small)
            _, ap_l = coco(gt_path, boxes, ids - small)
        out = {}
        for n in NAMES:
            s, l = ap_s.get(n, float('nan')), ap_l.get(n, float('nan'))
            out[n] = 100 * (l if s != s else s if l != l else shares[n] * s + (1 - shares[n]) * l)
        return out

    table = {}
    for name, tiers in RECIPES.items():
        boxes = hedge_recipe(predictions, tiers)
        table[name] = [per_class_mix(boxes, h) for h in halves]
        print(f'  {name:30s} mix {np.mean(list(table[name][0].values())):.2f} / '
              f'{np.mean(list(table[name][1].values())):.2f}', flush=True)
    current = 'current: sizes 8%, shifts 8%'
    gains = []
    for pick, report in ((0, 1), (1, 0)):
        chosen = {n: max(RECIPES, key=lambda r: table[r][pick][n]) for n in NAMES}
        score = np.mean([table[chosen[n]][report][n] for n in NAMES])
        ref = np.mean(list(table[current][report].values()))
        gains.append(score - ref)
        print(f'chosen on half {pick}: ' + ', '.join(f'{n}: {chosen[n]}' for n in NAMES))
        print(f'   -> half {report}: {score:.2f} vs current {ref:.2f} ({score - ref:+.2f})')
    print(f'cross-fitted gain of per-class recipes over the shared one: {np.mean(gains):+.2f}')


if __name__ == '__main__':
    main()
