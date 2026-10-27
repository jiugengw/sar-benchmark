"""Every labelled image holds objects of ONE class (its image label). So a box whose class differs from the image's
main class is almost always a false alarm, yet it competes in its class's ranking at full score. Does pushing such
boxes to the bottom of the ranking raise official-mix mAP? Same boxes, only their order changes; no GPU.

The image's main class comes from the detector itself: the class of its most confident box (or the class with the
largest summed score). Boxes of other classes keep their relative order but drop below every consistent box; with
--keep above a score ratio, a box stays in place when its score is at least that share of the main class's best.
Scored like group_runs.py (rounded + hedged, official-mix on the clean holdout), on the two twin-grouped halves.

    python tools/analysis/class_consistency_study.py results/ft40-s1r-flipmem-338727/one2many

Result on the flip test (338727, 27 Oct): official-mix 58.32 -> 58.33-58.35 (+0.01 to +0.03) for every variant; only
8 of 939 holdout images have confident boxes (score >= 0.25) of more than one class, so there is little to demote.
The same day, per-class hedge recipes gained +0.00 and a third hedge tier (+-16% sizes) +0.02 on this model.
"""
import contextlib, io, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo')
from official_mix_score import NAMES, official_shares
from resolution_probe import coco
from sarbench.twins import clean_image_ids
from hedge_study import CHOSEN_TIERS, hedge_tiers
from ranking_study import twin_groups
from round_boxes import round_predictions
results = Path(sys.argv[1])
gt_path = str(results / 'instances_val_holdout.json'); gt = json.load(open(gt_path))
images = {im['id']: im for im in gt['images']}; widths = {i: im['width'] for i, im in images.items()}
clean = set(clean_image_ids(gt['images']))
raw = json.load(open(results / 'predictions_val.json'))
shares = official_shares(Path('submissions/memorise-lr2-332326/memorise-lr2-332326.zip'), Path('data/course-data/val/images.json'))
groups = twin_groups(gt['images']); unique = sorted(set(groups.values()))
half = set(np.random.default_rng(0).permutation(unique)[: len(unique) // 2])
halves = [{i for i in clean if groups[i] in half}, {i for i in clean if groups[i] not in half}]

def mix(boxes, ids):
    small = {i for i in ids if widths[i] == 256}
    with contextlib.redirect_stdout(io.StringIO()):
        _, x = coco(gt_path, boxes, small); _, y = coco(gt_path, boxes, ids - small)
    nan = float('nan')
    return np.mean([100 * (y[n] if x.get(n, nan) != x.get(n, nan) else x[n] if y.get(n, nan) != y.get(n, nan)
                           else shares[n] * x[n] + (1 - shares[n]) * y[n]) for n in NAMES])

def main_class(boxes, how):
    per = defaultdict(lambda: defaultdict(float))
    for b in boxes:
        c = per[b['image_id']]
        c[b['category_id']] = max(c[b['category_id']], b['score']) if how == 'max' else c[b['category_id']] + b['score']
    return {i: max(c, key=c.get) for i, c in per.items()}, per

def demote(boxes, how, keep):
    top, per = main_class(boxes, how)
    best = {i: max(c.values()) for i, c in per.items()}
    out = []
    for b in boxes:
        b = dict(b)
        if b['category_id'] != top[b['image_id']] and not (keep and b['score'] >= keep * best[b['image_id']]):
            b['score'] *= 1e-4  # below every consistent box and its hedge copies, order kept
        out.append(b)
    return out

base = hedge_tiers(round_predictions(raw, widths, 'whole-pixel-classes'), CHOSEN_TIERS)
mixed = sum(len({b['category_id'] for b in raw if b['image_id'] == i and b['score'] >= 0.25}) > 1 for i in images)
print(f'{len(images)} holdout images; {mixed} have confident boxes (score >= 0.25) of more than one class')
print(f"{'variant':34s} {'half A':>7s} {'half B':>7s} {'all clean':>9s}")
rows = [('current (rounded + hedged)', base)]
for how in ('max', 'sum'):
    for keep in (0, 0.5, 0.8):
        # demotion is applied to the raw boxes before hedging, and again after (hedge copies inherit the class)
        boxes = demote(hedge_tiers(round_predictions(demote(raw, how, keep), widths, 'whole-pixel-classes'), CHOSEN_TIERS), how, keep)
        rows.append((f'demote other classes ({how}{f", keep >= {keep:.0%}" if keep else ""})', boxes))
for name, boxes in rows:
    print(f'{name:34s} {mix(boxes, halves[0]):7.2f} {mix(boxes, halves[1]):7.2f} {mix(boxes, clean):9.2f}', flush=True)
