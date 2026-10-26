"""Labelled images only (train + holdout, no test images): per class, how many images have another labelled image
that is their copy as is, and how many have one only after a left-right / up-down / 180-degree flip (64 px
thumbnails, Pearson >= 0.9, same size). Decides which classes --dataset-flips-classes flips.

Run from the repository root (needs data/course-data):

    python tools/analysis/flip_pairs_labelled.py
"""
import csv, json
from collections import Counter
import numpy as np
from PIL import Image
NAMES = ['aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil_tank', 'playground', 'ship', 'wind_turbine']
ims = json.load(open('data/course-data/train/instances.json'))['images']
label = {int(r['image_id']): int(r['label']) for r in csv.DictReader(open('data/course-data/train/labels.csv'))}
def th(group): return np.stack([np.asarray(Image.open('data/course-data/train/' + im['file_name']).convert('L').resize((64, 64), Image.BILINEAR), dtype=np.float32) for im in group])
def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s
same, flip, total = Counter(), Counter(), Counter()
for w in sorted({im['width'] for im in ims}):
    g = [im for im in ims if im['width'] == w]
    if len(g) < 2: continue
    X = th(g); P, sp = norm(X)
    has_same = np.zeros(len(g), bool); has_flip = np.zeros(len(g), bool)
    for k, f in enumerate([lambda a: a, lambda a: a[:, :, ::-1], lambda a: a[:, ::-1], lambda a: a[:, ::-1, ::-1]]):
        A, sa = norm(np.ascontiguousarray(f(X)))
        for s in range(0, len(g), 1000):
            C = A[s:s + 1000] @ P.T; C[sa[s:s + 1000] < 4] = 0; C[:, sp < 4] = 0
            for i in range(C.shape[0]): C[i, s + i] = 0  # not itself
            hit = C.max(1) >= 0.9
            if k == 0: has_same[s:s + 1000] |= hit
            else: has_flip[s:s + 1000] |= hit
    for i, im in enumerate(g):
        c = label[im['id']]; total[c] += 1; same[c] += has_same[i]; flip[c] += has_flip[i] and not has_same[i]
print(f"{'class':13s} {'images':>6s} {'copy as is':>11s} {'only flipped':>13s}")
for c, n in enumerate(NAMES):
    print(f'{n:13s} {total[c]:6d} {same[c]:6d} ({100*same[c]/max(1,total[c]):3.0f}%) {flip[c]:7d} ({100*flip[c]/max(1,total[c]):3.0f}%)')
