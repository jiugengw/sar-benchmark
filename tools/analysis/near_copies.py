"""How close does each test image get to its nearest labelled image (64 px thumbnails, Pearson, as is or flipped /
turned 180, any size against any size)? Same for clean holdout vs training, where labels can tell whether a near match
is the same scene: share of the holdout image's boxes matched (IoU >= 0.5, same class) by the near image's boxes.

Run from the repository root (needs data/course-data and the holdout results on the cluster):

    python tools/analysis/near_copies.py results/ft40-s1r-memorise-337666/one2many/instances_val_holdout.json
"""
import json, sys
import numpy as np
from PIL import Image
sys.path.insert(0, '.')
from sarbench.twins import clean_image_ids
tr = json.load(open('data/course-data/train/instances.json')); val = json.load(open('data/course-data/val/images.json'))['images']
hold = json.load(open(sys.argv[1])); hf = {im['file_name'] for im in hold['images']}; clean = set(clean_image_ids(hold['images']))
def th(root, ims): return np.stack([np.asarray(Image.open(f'{root}/' + im['file_name']).convert('L').resize((64, 64), Image.BILINEAR), dtype=np.float32) for im in ims])
def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s
F = [lambda a: a, lambda a: a[:, :, ::-1], lambda a: a[:, ::-1], lambda a: a[:, ::-1, ::-1]]
def best(root, tg, pool):
    P, sp = norm(th('data/course-data/train', pool)); X = th(root, tg); b = np.full(len(tg), -1.0); arg = np.zeros((len(tg), 2), int)
    for k, f in enumerate(F):
        A, sa = norm(np.ascontiguousarray(f(X))); C = A @ P.T; C[sa < 4] = -1; C[:, sp < 4] = -1
        m = C.max(1); better = m > b; b[better] = m[better]; arg[better] = np.stack([C.argmax(1), np.full(len(tg), k)], 1)[better]
    return b, arg
bins = [0.9, 0.8, 0.7, 0.6, 0.5, -2]
def hist(b):
    out, hi = [], 2
    for lo in bins:
        out.append(f'{">=" + str(lo) if lo > -1 else "<0.5"}: {((b >= lo) & (b < hi)).sum()}'); hi = lo
    return ', '.join(out)
b, _ = best('data/course-data/val', val, tr['images'])
print(f'test ({len(val)} images), best correlation with any labelled image: {hist(b)}')
pool = [im for im in tr['images'] if im['file_name'] not in hf]; tg = [im for im in hold['images'] if im['id'] in clean]
b, arg = best('data/course-data/train', tg, pool)
print(f'clean holdout ({len(tg)}), best correlation with any training image: {hist(b)}')
by = {}
for a in tr['annotations']: by.setdefault(a['image_id'], []).append(a)
def box(bb, k, W, H):
    x, y, w, h = bb
    if k in (1, 3): x = W - x - w
    if k in (2, 3): y = H - y - h
    return [x, y, w, h]
def iou(a, c):
    iw = max(0, min(a[0]+a[2], c[0]+c[2]) - max(a[0], c[0])); ih = max(0, min(a[1]+a[3], c[1]+c[3]) - max(a[1], c[1]))
    i = iw * ih; return i / (a[2]*a[3] + c[2]*c[3] - i + 1e-9)
for lo, hi in [(0.9, 2), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6)]:
    shares = []
    for i in np.nonzero((b >= lo) & (b < hi))[0]:
        im, src, k = tg[i], pool[arg[i, 0]], arg[i, 1]
        if im['width'] != src['width']: continue
        mine = by.get(im['id'], []); theirs = [box(a['bbox'], k, src['width'], src['height']) + [a['category_id']] for a in by.get(src['id'], [])]
        if mine: shares.append(np.mean([any(t[4] == a['category_id'] and iou(a['bbox'], t[:4]) >= 0.5 for t in theirs) for a in mine]))
    print(f'  holdout images with best correlation {lo}-{min(hi, 1):.1f}: same-size pairs {len(shares)}, '
          f'their boxes found in the near image (flipped back, IoU>=0.5): {100*np.mean(shares) if shares else float("nan"):.0f}%')
