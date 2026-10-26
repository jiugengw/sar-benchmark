"""Exact tile-grid overlap (512 px tiles every 312 px, so neighbours share 200 px) between images, on 64 x 64
thumbnails (8 px cells, shift 39 cells). A neighbour = correlation >= 0.95 over the shared strip or corner.
A. clean holdout 512 px images vs training images outside the holdout: area covered, labels inside, found rates.
B. Codabench test 512 px images vs all labelled 512 px images: how many have a labelled neighbour, area covered,
   and how many labelled boxes of each class fall inside test images (counted only, nothing written).

Run from the repository root (needs data/course-data and the holdout results on the cluster):

    python tools/analysis/tile_grid_overlap.py results/ft40-s1r-memorise-337666/one2many
"""
import json, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '.')
from sarbench.twins import clean_image_ids
R = Path(sys.argv[1]); troot, vroot = Path('data/course-data/train'), Path('data/course-data/val')
hold = json.loads((R / 'instances_val_holdout.json').read_text())
train = json.loads((troot / 'instances.json').read_text())
val = json.loads((vroot / 'images.json').read_text())['images']
names = {c['id']: c['name'] for c in train['categories']}
clean = set(clean_image_ids(hold['images'])); hold_files = {im['file_name'] for im in hold['images']}
N, S, W = 64, 39, 25
def thumbs(root, ims):
    return np.stack([np.asarray(Image.open(root / im['file_name']).convert('L').resize((N, N), Image.BILINEAR), dtype=np.float32) for im in ims])
def rng(o, target):
    return (S, N) if (o == 1) == target and o else (0, W) if o else (0, N)
def neighbours(T, P):
    """For each target row: list of (pool index, ox, oy, corr) with corr >= 0.95 over the shared region."""
    out = defaultdict(list)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            if not ox and not oy: continue
            (tr0, tr1), (tc0, tc1) = rng(oy, True), rng(ox, True)
            (pr0, pr1), (pc0, pc1) = rng(-oy, True), rng(-ox, True)
            a = T[:, tr0:tr1, tc0:tc1].reshape(len(T), -1); b = P[:, pr0:pr1, pc0:pc1].reshape(len(P), -1)
            sa, sb = a.std(1), b.std(1)
            a = (a - a.mean(1, keepdims=True)) / (np.linalg.norm(a - a.mean(1, keepdims=True), axis=1, keepdims=True) + 1e-6)
            b = (b - b.mean(1, keepdims=True)) / (np.linalg.norm(b - b.mean(1, keepdims=True), axis=1, keepdims=True) + 1e-6)
            C = a @ b.T
            C[sa < 4] = 0; C[:, sb < 4] = 0  # flat strips (no-data borders) match anything
            for i, j in zip(*np.nonzero(C >= 0.95)):
                out[i].append((j, ox, oy, float(C[i, j])))
    return out
def covered(nb):
    m = np.zeros((512, 512), bool)
    for _, ox, oy, _ in nb:
        r = slice(312, 512) if oy == 1 else slice(0, 200) if oy == -1 else slice(0, 512)
        c = slice(312, 512) if ox == 1 else slice(0, 200) if ox == -1 else slice(0, 512)
        m[r, c] = True
    return m
def iou(a, b):
    iw = max(0, min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])); ih = max(0, min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    i = iw * ih; return i / (a[2]*a[3] + b[2]*b[3] - i + 1e-9)

# A. holdout
pool_ims = [im for im in train['images'] if im['width'] == 512 and im['file_name'] not in hold_files]
targ_ims = [im for im in hold['images'] if im['width'] == 512 and im['id'] in clean]
P, T = thumbs(troot, pool_ims), thumbs(troot, targ_ims)
nb = neighbours(T, P)
cov = {targ_ims[i]['id']: covered(nb.get(i, [])) for i in range(len(targ_ims))}
print(f'A. clean holdout 512 px images: {len(targ_ims)}; with a training neighbour {sum(1 for i in range(len(targ_ims)) if nb.get(i))}; '
      f'area covered by training tiles {100*np.mean([m.mean() for m in cov.values()]):.0f}%', flush=True)
print('   directions:', Counter((ox, oy) for v in nb.values() for _, ox, oy, _ in v).most_common())
preds = defaultdict(list)
for p in json.loads((R / 'predictions_val.json').read_text()): preds[(p['image_id'], p['category_id'])].append(p['bbox'])
res = defaultdict(lambda: defaultdict(list))
for a in hold['annotations']:
    if a['image_id'] not in cov: continue
    x, y, w, h = a['bbox']; inside = cov[a['image_id']][min(511, int(y + h / 2)), min(511, int(x + w / 2))]
    found = max([iou(a['bbox'], b) for b in preds[(a['image_id'], a['category_id'])]] or [0]) >= 0.5
    res[names[a['category_id']]]['inside' if inside else 'outside'].append(found)
for n, g in sorted(res.items()):
    i, o = g['inside'], g['outside']
    print(f'   {n:13s} labels inside training-covered area {len(i):4d} of {len(i)+len(o):4d} '
          f'({100*len(i)/max(1,len(i)+len(o)):3.0f}%)  found inside {100*np.mean(i) if i else float("nan"):3.0f}%  outside {100*np.mean(o) if o else float("nan"):3.0f}%')

# B. test
pool_ims = [im for im in train['images'] if im['width'] == 512]
targ_ims = [im for im in val if im['width'] == 512]
P, T = thumbs(troot, pool_ims), thumbs(vroot, targ_ims)
nb = neighbours(T, P)
covs = [covered(nb.get(i, [])) for i in range(len(targ_ims))]
print(f'\nB. Codabench test 512 px images: {len(targ_ims)}; with a labelled neighbour {sum(1 for i in range(len(targ_ims)) if nb.get(i))}; '
      f'area covered by labelled tiles {100*np.mean([m.mean() for m in covs]):.0f}%')
print('   directions:', Counter((ox, oy) for v in nb.values() for _, ox, oy, _ in v).most_common())
by_img = defaultdict(list)
for a in train['annotations']: by_img[a['image_id']].append(a)
cnt = Counter()
for i, lst in nb.items():
    seen = set()
    for j, ox, oy, _ in lst:
        for a in by_img[pool_ims[j]['id']]:
            x, y, w, h = a['bbox']; cx, cy = x + w / 2 + ox * 312, y + h / 2 + oy * 312
            key = (a['category_id'], round(cx / 4), round(cy / 4))
            if 0 <= cx < 512 and 0 <= cy < 512 and key not in seen:
                seen.add(key); cnt[names[a['category_id']]] += 1
print('   labelled boxes whose centre falls inside a test image (deduplicated):', dict(cnt.most_common()))
