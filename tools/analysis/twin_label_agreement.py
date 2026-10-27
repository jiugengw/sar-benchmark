"""Do holdout twins carry the same boxes as their training copy? Twins (same capture, other brightness/sharpness)
score only ~85 mAP after memorise. If their labels match the training copy's, the gap is the model failing to
recognise the re-processed pixels (what the brightness + sharpness jitter test targets); if the labels differ, the
gap is annotation noise and no model can close it.

Analysis only: scores the training copy's boxes as if they were predictions on the twin (label copying is not used
for any submission). Run from the repository root:

    python tools/analysis/twin_label_agreement.py results/yolo26m-100-p2-328339/seed_0/instances_val_holdout.json \\
        [results/twin-copies.json]

The optional second argument saves {twin image id: training copy id} for train_copy_ap.py.
"""
import json, sys
from collections import defaultdict
import numpy as np
from PIL import Image
sys.path.insert(0, '.')
from sarbench.twins import clean_image_ids
from sarbench.metrics import detection_metrics
gt_path = sys.argv[1]
gt = json.load(open(gt_path)); train = json.load(open('data/course-data/train/instances.json'))
hf = {im['file_name'] for im in gt['images']}; clean = set(clean_image_ids(gt['images']))
boxes = defaultdict(list)
for a in train['annotations']:
    boxes[a['image_id']].append(a)
def load(im, n):
    return np.asarray(Image.open('data/course-data/train/' + im['file_name']).convert('L').resize((n, n), Image.BILINEAR), dtype=np.float32)
def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)
match = {}
for w in sorted({im['width'] for im in gt['images']}):
    tg = [im for im in gt['images'] if im['width'] == w and im['id'] not in clean]
    pool = [im for im in train['images'] if im['width'] == w and im['file_name'] not in hf]
    if not tg or not pool: continue
    P, sp = norm(np.stack([load(im, 64) for im in pool])); A, sa = norm(np.stack([load(im, 64) for im in tg]))
    C = A @ P.T; C[sa < 4] = 0; C[:, sp < 4] = 0
    for i, im in enumerate(tg):
        j = int(C[i].argmax())
        if C[i, j] >= 0.9: match[im['id']] = (pool[j]['id'], float(C[i, j]))
twins = sorted(match)
if len(sys.argv) > 2:
    json.dump({t: match[t][0] for t in twins}, open(sys.argv[2], 'w'))
copied, best_iou, extra = [], [], 0
for t in twins:
    src = boxes[match[t][0]]
    copied += [dict(image_id=t, category_id=a['category_id'], bbox=a['bbox'], score=1.0) for a in src]
    for g in boxes[t]:
        best_iou.append(max([iou(g['bbox'], a['bbox']) for a in src if a['category_id'] == g['category_id']] or [0.0]))
    extra += max(0, len(src) - len(boxes[t]))
best_iou = np.array(best_iou)
names = {c['id']: c['name'] for c in gt['categories']}
print(f'twins matched to a training copy: {len(twins)} of {sum(1 for im in gt["images"] if im["id"] not in clean)} '
      f'(pixel r median {np.median([m[1] for m in match.values()]):.3f})')
print(f'twin boxes: {len(best_iou)}; same-class box in the training copy at IoU >= 0.95: {np.mean(best_iou >= 0.95):.1%}, '
      f'>= 0.75: {np.mean(best_iou >= 0.75):.1%}, >= 0.5: {np.mean(best_iou >= 0.5):.1%}, none: {np.mean(best_iou < 0.1):.1%}; '
      f'extra boxes in the copies: {extra}')
report = detection_metrics(gt_path, copied, twins)
print(f"training copy's boxes scored as predictions on the twins: mAP {100 * report['mAP']:.1f}"
      + ''.join(f"; {k} {100 * v:.1f}" for k, v in report.items() if k in ('AP50', 'AP75')))
if 'per_class_AP' in report:
    print('per class: ' + ', '.join(f'{names.get(k, k)} {100 * v:.1f}' for k, v in report['per_class_AP'].items()))
