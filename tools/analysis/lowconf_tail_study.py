"""Below the submission's score floor: do the detector's own boxes under 0.001 add mAP when ranked last?

New-scene bridges and playgrounds are mostly missed outright (no box at IoU 0.1 above score 0.05), and the submission
writer drops every box under 0.001. Boxes appended at the bottom of a class's ranking can never lower its AP, so the
detector's sub-0.001 boxes are free recall if any of them land on a missed object. Same holdout run, predicted again
with conf 1e-5 and up to 1000 boxes per image (make_submission.py --conf --max-det); its boxes >= 0.001 are the
submission's boxes.

Ranking bands, each wholly below the one before: originals (>= 0.001), then the two hedge tiers (hedge_study.py),
with the low boxes placed last, between the tiers, or straight after the originals. Every image and class keeps at
most 100 boxes (the scorer's limit), filled band by band. Scored like group_runs.py (rounded, official-mix on the
clean holdout) on the two twin-grouped halves, plus per class and on new scenes.

    python tools/analysis/lowconf_tail_study.py results/tail-lowconf-340514
"""
import contextlib, io, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo')
from official_mix_score import NAMES, official_shares
from resolution_probe import coco
from sarbench.twins import clean_image_ids
from hedge_study import SIZES, SHIFTS
from ranking_study import twin_groups
from round_boxes import round_predictions
results = Path(sys.argv[1])
gt_path = str(results / 'instances_val_holdout.json'); gt = json.load(open(gt_path))
images = {im['id']: im for im in gt['images']}; widths = {i: im['width'] for i, im in images.items()}
clean = set(clean_image_ids(gt['images']))
raw = round_predictions(json.load(open(results / 'predictions_lowconf.json')), widths, 'whole-pixel-classes')
originals = [p for p in raw if p['score'] >= 0.001]; low = [p for p in raw if p['score'] < 0.001]
shares = official_shares(Path('submissions/memorise-lr2-332326/memorise-lr2-332326.zip'), Path('data/course-data/val/images.json'))
groups = twin_groups(gt['images']); unique = sorted(set(groups.values()))
half = set(np.random.default_rng(0).permutation(unique)[: len(unique) // 2])
halves = [{i for i in clean if groups[i] in half}, {i for i in clean if groups[i] not in half}]
new = set(json.load(open(results / 'new_scene_ids.json'))) if (results / 'new_scene_ids.json').exists() else None

def per_class(boxes, ids):
    small = {i for i in ids if widths[i] == 256}
    with contextlib.redirect_stdout(io.StringIO()):
        _, x = coco(gt_path, boxes, small); _, y = coco(gt_path, boxes, ids - small)
    nan = float('nan')
    return {n: 100 * (y[n] if x.get(n, nan) != x.get(n, nan) else x[n] if y.get(n, nan) != y.get(n, nan)
                      else shares[n] * x[n] + (1 - shares[n]) * y[n]) for n in NAMES}

def copies(boxes, transforms):
    out = []
    for p in boxes:
        x, y, w, h = p['bbox']; cx, cy = x + w / 2, y + h / 2
        for sw, sh, *shift in transforms:
            nw, nh = w * sw, h * sh; dx, dy = (shift[0] * w, shift[1] * h) if shift else (0.0, 0.0)
            out.append({**p, 'bbox': [cx + dx - nw / 2, cy + dy - nh / 2, nw, nh]})
    return out

def ranked(bands):
    """bands: lists of boxes, best first. Band k > 0 keeps its order with scores in (1e-3, 1] x 1e-3 x 1e-4^(k-1)
    (relative score: the box's own for originals, /0.001 for low boxes), so every band ranks below the one before."""
    room = defaultdict(lambda: 100); out = []
    for k, band in enumerate(bands):
        for p in sorted(band, key=lambda p: -p['score']):
            key = (p['image_id'], p['category_id'])
            if room[key] <= 0: continue
            room[key] -= 1
            rel = p['score'] / 0.001 if p['score'] < 0.001 else p['score']
            out.append({**p, 'score': p['score'] if k == 0 else rel * 1e-3 * 1e-4 ** (k - 1)})
    return out

t1, t2 = copies(originals, SIZES), copies(originals, SHIFTS)
print(f'{len(originals)} boxes >= 0.001, {len(low)} below (min score {min(p["score"] for p in raw):.1e})')
variants = {'current: originals, tier 1, tier 2': [originals, t1, t2],
            'low boxes last': [originals, t1, t2, low],
            'low boxes between the tiers': [originals, t1, low, t2],
            'low boxes right after originals': [originals, low, t1, t2]}
base = None
show = ['bridge', 'playground', 'harbor', 'ship']
print(f"{'variant':34s} {'half A':>7s} {'half B':>7s} {'clean':>7s}  " + ' '.join(f'{n[:8]:>8s}' for n in show) + (f" {'new':>6s}" if new else ''))
for name, bands in variants.items():
    boxes = ranked(bands)
    a, b, c = (per_class(boxes, h) for h in (*halves, clean))
    row = [np.nanmean(list(v.values())) for v in (a, b, c)]
    extra = f' {np.nanmean(list(per_class(boxes, new & clean).values())):6.2f}' if new else ''
    if base is None: base = (row, c)
    d = '' if name.startswith('current') else f'   gain {row[0] - base[0][0]:+.2f} / {row[1] - base[0][1]:+.2f} / {row[2] - base[0][2]:+.2f}'
    print(f'{name:34s} {row[0]:7.2f} {row[1]:7.2f} {row[2]:7.2f}  ' + ' '.join(f'{c[n]:8.2f}' for n in show) + extra + d, flush=True)
