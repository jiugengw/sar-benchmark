"""Re-rank boxes by localisation certainty from the detector's own suppressed candidates (needs raw_candidates.json).

For each submitted box, its "spread" is 1 - the mean IoU between it and the detector's other candidates of the same
class that overlap it (IoU >= 0.5, score >= 5% of its own): candidates that agree closely mean a well-pinned box.
On the holdout the spread predicts a box's IoU with its label about as strongly as its confidence does (Spearman
-0.42 vs +0.49, in every class). New score = score * (1 - spread) ** gamma; boxes are the same, only the order
changes; the chosen box hedging (hedge_study.py) is applied on top. gamma is chosen on one holdout half and scored on
the other (official-mix mAP).

    python models/yolo/spread_rerank_study.py results/ft40-m4rep-bp4-memorise-335138/one2many
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


def spreads(submitted: list[dict], raw: list[dict], min_iou: float = 0.5, rel_score: float = 0.05) -> list[float]:
    from tail_study import iou_one
    by_key = defaultdict(list)
    for c in raw:
        by_key[(c['image_id'], c['category_id'])].append(c)
    out = []
    for p in submitted:
        ious = [v for c in by_key[(p['image_id'], p['category_id'])]
                if c['score'] >= rel_score * p['score'] and c['bbox'] != p['bbox']
                and (v := iou_one(c['bbox'], p['bbox'])) >= min_iou]
        out.append(1 - float(np.mean(ious)) if ious else 0.0)
    return out


def rerank(submitted: list[dict], spread: list[float], gamma: float) -> list[dict]:
    """Same boxes, new scores, rescaled so the lowest stays >= 0.001 (copies must still rank below every box)."""
    scores = np.array([p['score'] * (1 - s) ** gamma for p, s in zip(submitted, spread)])
    scores = 0.001 + (1 - 0.001) * (scores - scores.min()) / max(scores.max() - scores.min(), 1e-12)
    return [{**p, 'score': float(s)} for p, s in zip(submitted, scores)]


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

    spread = spreads(submitted, raw)
    rows = []
    for gamma in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        boxes = hedge_tiers(rerank(submitted, spread, gamma), CHOSEN_TIERS)
        scores = [mix(boxes, h) for h in halves]
        rows.append((gamma, scores))
        print(f'  gamma {gamma:4.1f}: official-mix with hedging {scores[0]:.2f} / {scores[1]:.2f}', flush=True)
    base = rows[0][1]
    gains = []
    for pick, report in ((0, 1), (1, 0)):
        best = max(rows, key=lambda r: r[1][pick])
        gains.append(best[1][report] - base[report])
        print(f'gamma chosen on half {pick}: {best[0]} -> half {report}: {best[1][report] - base[report]:+.2f}')
    print(f'cross-fitted gain over hedging alone: {np.mean(gains):+.2f}')


if __name__ == '__main__':
    main()
