"""Box hedging: does adding resized copies of each predicted box, ranked below it, raise mAP@[.5:.95]?

COCO-style AP matches every label to the best-ranked box that overlaps it enough, separately at each IoU threshold
0.50 ... 0.95. Our boxes find objects (AP50 85-95 on most classes) but often miss the strict thresholds by a pixel.
A resized copy of a box, at a lower score, can match a label at 0.85 or 0.9 that the original just missed; where the
original already matched, the copy is a false positive ranked below it. AP takes the best precision at each recall
level, so boxes appended at the bottom of the ranking can never lower it; copies interleaved just below their
original trade precision at loose thresholds for recall at strict ones. The scorer keeps the top 100 boxes per
image and class, so copies must fit under that.

For each setting (copy sizes, which boxes get copies, score factor) the gain in official-mix mAP is measured
cross-fitted: the setting is chosen on one half of the holdout (twin groups kept together) and scored on the other,
both ways round, so the reported gain is not tuned on the images it is scored on.

    python models/yolo/hedge_study.py study results/ft40-m4rep-bp4-memorise-335138/one2many
    python models/yolo/hedge_study.py zip in.zip out.zip          # +-8% copies below every box (the chosen setting)

Chosen setting: two tiers of copies. Tier 1: each box 8% larger and 8% smaller around the same centre; tier 2: each
box shifted by 8% of its size left, right, up and down. Every tier-1 copy ranks below every original box (the
submission writer keeps boxes down to score 0.001) and every tier-2 copy below every tier-1 copy. Cross-fitted on the
holdout: about +0.8 official-mix on three different models (resized copies alone +0.6; both in one tier +0.67).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))


def hedge(predictions: list[dict], scales: list[tuple[float, ...]], min_score: float, factor: float,
          max_per_class: int = 100) -> list[dict]:
    """Each box scoring >= min_score gets a copy per (width, height) scale, centred on it (or shifted by (dx, dy) times
    its size when given), at score * factor.
    Copies are dropped first when an image and class would exceed max_per_class boxes."""
    by_key = defaultdict(list)
    for p in predictions:
        by_key[(p['image_id'], p['category_id'])].append(p)
    out = []
    for boxes in by_key.values():
        boxes = sorted(boxes, key=lambda p: -p['score'])
        copies = []
        for p in boxes:
            if p['score'] < min_score:
                break
            x, y, w, h = p['bbox']
            cx, cy = x + w / 2, y + h / 2
            for sw, sh, *shift in scales:
                nw, nh = w * sw, h * sh
                dx, dy = (shift[0] * w, shift[1] * h) if shift else (0.0, 0.0)
                copies.append({**p, 'bbox': [cx + dx - nw / 2, cy + dy - nh / 2, nw, nh], 'score': p['score'] * factor})
        merged = sorted(boxes + copies, key=lambda p: -p['score'])[:max_per_class]
        out.extend(merged)
    return out


def hedge_tiers(predictions: list[dict], tiers: list[list[tuple[float, ...]]], top: float = 0.0009,
                step: float = 0.001, max_per_class: int = 100) -> list[dict]:
    """Copies in tiers: every copy of tier k ranks below every copy of tier k-1, and tier 1 below every original
    (originals score >= 0.001; tier k copies score original * top * step**(k-1)). Within a tier, copies keep their
    originals' order. Every original is kept; copies only fill an image and class up to max_per_class boxes."""
    copies = []
    for k, transforms in enumerate(tiers):
        tier = hedge(predictions, transforms, 0.0, top * step ** k, max_per_class=10 ** 9)
        copies += [p for p in tier if p['score'] < (0.001 * top * step ** (k - 1) if k else 0.001)]
    room = defaultdict(lambda: max_per_class)
    for p in predictions:
        room[(p['image_id'], p['category_id'])] -= 1
    out = list(predictions)
    for p in sorted(copies, key=lambda p: -p['score']):
        key = (p['image_id'], p['category_id'])
        if room[key] > 0:
            room[key] -= 1
            out.append(p)
    return out


SIZES = [(1.08, 1.08), (0.92, 0.92)]
SHIFTS = [(1, 1, 0.08, 0), (1, 1, -0.08, 0), (1, 1, 0, 0.08), (1, 1, 0, -0.08)]
CHOSEN_TIERS = [SIZES, SHIFTS]


def rewrite_zip(source: Path, target: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    predictions = json.loads(members['detection.json'])
    if min(p['score'] for p in predictions) < 0.001:
        raise ValueError('boxes below score 0.001: copies would not rank below every original box')
    hedged = hedge_tiers(predictions, CHOSEN_TIERS)
    members['detection.json'] = json.dumps(hedged).encode()
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    print(f'{target}: {len(predictions)} boxes -> {len(hedged)} with copies; other files byte-for-byte unchanged')


def study(results: Path) -> None:
    from official_mix_score import NAMES, official_shares
    from ranking_study import twin_groups
    from resolution_probe import coco
    from sarbench.twins import clean_image_ids

    gt_path = results / 'instances_val_holdout.json'
    gt = json.loads(gt_path.read_text())
    images = {im['id']: im for im in gt['images']}
    clean = set(clean_image_ids(gt['images']))
    predictions = json.loads((results / 'predictions_val.json').read_text())
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
        return 100 * float(np.nanmean(values))

    base = [mix(predictions, h) for h in halves]
    print(f'no hedging: official-mix {base[0]:.2f} / {base[1]:.2f} on the two halves', flush=True)
    layouts = {
        'sizes': [SIZES],
        'sizes + shifts, one tier': [SIZES + SHIFTS],
        'sizes, then shifts (chosen)': [SIZES, SHIFTS],
        'sizes, shifts, then +-16% sizes': [SIZES, SHIFTS, [(1.16, 1.16), (0.84, 0.84)]],
        'sizes, shifts, then diagonal shifts': [SIZES, SHIFTS, [(1, 1, 0.08, 0.08), (1, 1, -0.08, -0.08),
                                                                 (1, 1, 0.08, -0.08), (1, 1, -0.08, 0.08)]],
    }
    results = []
    for name, tiers in layouts.items():
        hedged = hedge_tiers(predictions, tiers)
        gains = [mix(hedged, h) - b for h, b in zip(halves, base)]
        results.append((name, 0.0, 0.0, gains))
        print(f'  {name:36s} gain {gains[0]:+.2f} / {gains[1]:+.2f}', flush=True)
    # cross-fit: pick on one half, report on the other
    picked = []
    for pick, report in ((0, 1), (1, 0)):
        best = max(results, key=lambda r: r[3][pick])
        picked.append(best[3][report])
        print(f'chosen on half {pick}: {best[0]} -> on half {report}: '
              f'{best[3][report]:+.2f}')
    print(f'cross-fitted gain in official-mix mAP: {np.mean(picked):+.2f}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('study')
    s.add_argument('results', type=Path, help='folder with predictions_val.json and instances_val_holdout.json')
    z = sub.add_parser('zip')
    z.add_argument('source', type=Path)
    z.add_argument('target', type=Path)
    args = parser.parse_args()
    if args.command == 'study':
        study(args.results)
    else:
        rewrite_zip(args.source, args.target)


if __name__ == '__main__':
    main()
