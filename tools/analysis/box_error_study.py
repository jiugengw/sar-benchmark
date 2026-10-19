"""Where do one class's boxes go wrong? Clean holdout, memorise-lr2 model, NMS 0.7 (as submitted).

For every clean-holdout label of the class: best IoU with our boxes (score >= 0.1), centre offset, width and height
error, split by label size, image size, whole-pixel vs fractional labels, boxes per image, image class and shape.
Writes results/pp-memorise-lr2/box_errors_<class id>.json for tools/analysis/worst_box_crops.py.

    python tools/analysis/box_error_study.py 8      # 8 = ship (COCO category id)
"""

import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'models/yolo'))
from postprocess_study import apply, iou_matrix
from sarbench.twins import clean_image_ids

raw = ROOT / 'results/pp-memorise-lr2/raw'
gt = json.loads((raw / 'instances_val_holdout.json').read_text())
images = {im['id']: im for im in gt['images']}
clean = set(clean_image_ids(gt['images']))
preds = apply(json.loads((raw / 'predictions_val.json').read_text()), 'nms', 0.7)
with (ROOT / 'data/course-data/train/labels.csv').open() as s:
    image_class = {int(r['image_id']): int(r['label']) for r in csv.DictReader(s)}
names = {c['id']: c['name'] for c in gt['categories']}
CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 8

by_image = defaultdict(list)
for p in preds:
    if p['category_id'] == CLASS and p['score'] >= 0.1:
        by_image[p['image_id']].append(p)

def xyxy(b):
    return np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], dtype=float)

rows = []
for a in gt['annotations']:
    if a['category_id'] != CLASS or a['image_id'] not in clean:
        continue
    g = xyxy(a['bbox'])
    cand = by_image.get(a['image_id'], [])
    best, best_iou = None, 0.0
    if cand:
        ious = iou_matrix(g[None], np.stack([xyxy(p['bbox']) for p in cand]))[0]
        k = int(np.argmax(ious)); best, best_iou = cand[k], float(ious[k])
    w, h = a['bbox'][2], a['bbox'][3]
    r = dict(image=a['image_id'], size=images[a['image_id']]['width'], w=w, h=h, iou=best_iou,
             integer=all(abs(v - round(v)) < 1e-3 for v in a['bbox']), image_class=image_class.get(a['image_id']),
             n_in_image=sum(1 for b in gt['annotations'] if b['image_id'] == a['image_id'] and b['category_id'] == CLASS))
    if best is not None and best_iou > 0.3:
        pb = best['bbox']
        r.update(dx=(pb[0] + pb[2] / 2) - (a['bbox'][0] + w / 2), dy=(pb[1] + pb[3] / 2) - (a['bbox'][1] + h / 2),
                 lw=np.log(pb[2] / w), lh=np.log(pb[3] / h), score=best['score'], pred=pb)
    rows.append(r)

def summary(sel, label):
    if not sel:
        return
    iou = np.array([r['iou'] for r in sel])
    m = [r for r in sel if 'dx' in r]
    dx = np.array([r['dx'] for r in m]); dy = np.array([r['dy'] for r in m])
    lw = np.array([r['lw'] for r in m]); lh = np.array([r['lh'] for r in m])
    print(f"{label:28s} n {len(sel):4d}  found {100*np.mean(iou>0.5):5.1f}%  IoU>=.75 {100*np.mean(iou>=.75):5.1f}%  "
          f">=.85 {100*np.mean(iou>=.85):5.1f}%  med IoU {np.median(iou):.2f}  |dx| {np.median(abs(dx)):.2f} "
          f"|dy| {np.median(abs(dy)):.2f}  w err {100*np.median(abs(lw)):4.1f}%  h err {100*np.median(abs(lh)):4.1f}%  "
          f"w bias {100*np.median(lw):+4.1f}% h bias {100*np.median(lh):+4.1f}%")

print(f'== {names[CLASS]} on clean holdout: {len(rows)} labels, {len(clean)} images')
summary(rows, 'all')
for lo, hi in ((0, 8), (8, 12), (12, 16), (16, 24), (24, 40), (40, 1e9)):
    summary([r for r in rows if lo <= np.sqrt(r['w'] * r['h']) < hi], f'size {lo}-{hi:g} px')
for s in (512, 256):
    summary([r for r in rows if r['size'] == s], f'{s}-px images')
summary([r for r in rows if r['integer']], 'whole-pixel labels')
summary([r for r in rows if not r['integer']], 'fractional labels')
summary([r for r in rows if r['n_in_image'] == 1], 'alone in image')
summary([r for r in rows if 2 <= r['n_in_image'] <= 5], '2-5 in image')
summary([r for r in rows if r['n_in_image'] > 5], '>5 in image')
for c in sorted({r['image_class'] for r in rows}):
    summary([r for r in rows if r['image_class'] == c], f'image class {c}')
summary([r for r in rows if r['w'] > 1.5 * r['h']], 'wide (w>1.5h)')
summary([r for r in rows if r['h'] > 1.5 * r['w']], 'tall (h>1.5w)')
summary([r for r in rows if max(r['w'], r['h']) <= 1.5 * min(r['w'], r['h'])], 'squarish')
sizes = np.array([np.sqrt(r['w'] * r['h']) for r in rows])
print('label size percentiles (sqrt area, px):', np.percentile(sizes, [10, 25, 50, 75, 90]).round(1).tolist())
print('IoU of a box shifted by 1 px both ways, at the median size:',
      round(float(iou_matrix(np.array([[0, 0, np.median(sizes), np.median(sizes)]]),
                             np.array([[1, 1, np.median(sizes) + 1, np.median(sizes) + 1]]))[0, 0]), 3))
json.dump(rows, open(raw.parent / f'box_errors_{CLASS}.json', 'w'), default=float)
