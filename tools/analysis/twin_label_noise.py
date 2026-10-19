"""How consistently are the same scenes labelled? Compare the boxes of near-identical twin training images
(sarbench/train_twins.json): share of pairs with identical boxes, and for the rest, box IoU between the two labels;
then, per pair, the shared shift of all boxes (image registration) vs box-by-box scatter (labelling).

    python tools/analysis/twin_label_noise.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
coco = json.load(open(ROOT / 'data/course-data/train/instances.json'))
pairs = json.load(open(ROOT / 'sarbench/train_twins.json'))['pairs']
byfile = {im['file_name']: im['id'] for im in coco['images']}
ann = defaultdict(list)
for a in coco['annotations']:
    ann[a['image_id']].append(a)
names = {c['id']: c['name'] for c in coco['categories']}
def iou(a, b):
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih; return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)
stats = defaultdict(lambda: dict(pairs=0, identical=0, ious=[], unmatched=0, boxes=0))
for p in pairs:
    a, b = (byfile[f] for f in p['images'])
    A, B = ann[a], ann[b]
    if not A:
        continue
    s = stats[names[A[0]['category_id']]]; s['pairs'] += 1
    if sorted(tuple(round(v, 2) for v in x['bbox']) for x in A) == sorted(tuple(round(v, 2) for v in x['bbox']) for x in B):
        s['identical'] += 1; continue
    used = set()
    for x in A:
        best, bj = 0, None
        for j, y in enumerate(B):
            if j not in used and iou(x['bbox'], y['bbox']) > best:
                best, bj = iou(x['bbox'], y['bbox']), j
        s['boxes'] += 1
        if bj is not None and best > 0.3:
            used.add(bj); s['ious'].append(best)
        else:
            s['unmatched'] += 1
print(f"{'class':13s} {'pairs':>5s} {'same boxes':>10s} | differing pairs: {'boxes':>5s} {'IoU median':>10s} {'IoU<0.85':>8s} {'no match':>8s}")
for k, s in sorted(stats.items()):
    io = np.array(s['ious'])
    print(f"{k:13s} {s['pairs']:5d} {100 * s['identical'] / s['pairs']:9.0f}% | {s['pairs'] - s['identical']:5d} pairs {s['boxes']:5d} "
          f"{np.median(io) if len(io) else float('nan'):10.3f} {100 * np.mean(io < 0.85) if len(io) else float('nan'):7.0f}% "
          f"{100 * s['unmatched'] / max(1, s['boxes']):7.0f}%")

# Is the disagreement a shift of the whole image (registration) or box-by-box (labelling)?
print('\nper differing pair: shared shift of all boxes vs box-by-box scatter left after removing it (px, median)')
for cls in ('aircraft', 'harbor', 'oil_tank'):
    shared, scatter, size_err, ious_after = [], [], [], []
    for p in pairs:
        a, b = (byfile[f] for f in p['images'])
        A, B = ann[a], ann[b]
        if not A or names[A[0]['category_id']] != cls:
            continue
        if sorted(tuple(round(v, 2) for v in x['bbox']) for x in A) == sorted(tuple(round(v, 2) for v in x['bbox']) for x in B):
            continue
        m = []
        for x in A:
            y = max(B, key=lambda y: iou(x['bbox'], y['bbox']))
            if iou(x['bbox'], y['bbox']) > 0.3:
                m.append((x['bbox'], y['bbox']))
        if len(m) < 2:
            continue
        d = np.array([[(y[0] + y[2] / 2) - (x[0] + x[2] / 2), (y[1] + y[3] / 2) - (x[1] + x[3] / 2)] for x, y in m])
        shift = d.mean(0)
        shared.append(np.hypot(*shift)); scatter.extend(np.hypot(*(d - shift).T))
        for x, y in m:
            size_err.append(abs(np.log(y[2] / x[2])) / 2 + abs(np.log(y[3] / x[3])) / 2)
            ious_after.append(iou(x, [y[0] - shift[0], y[1] - shift[1], y[2], y[3]]))
    print(f'{cls:10s} pairs {len(shared):4d}  shared shift {np.median(shared):.2f}  scatter {np.median(scatter):.2f}  '
          f'size disagreement {100 * np.median(size_err):.1f}%  IoU after removing the shift: median {np.median(ious_after):.3f}, '
          f'<0.85 {100 * np.mean(np.array(ious_after) < 0.85):.0f}%')
