"""Distinct test images that repeat labelled content, split into what the current model already sees the right way
round and what only the flip test adds. Relations: whole-image copy (r >= 0.9), near-copy (r 0.7-0.9), 200 px strip
of a 312 px-grid neighbour (r >= 0.95); each as is or after flipping the test image (lr / ud / 180).

Run from the repository root (needs data/course-data and the holdout results on the cluster):

    python tools/analysis/test_copies_coverage.py <submission.zip>   # classes from its classification.csv
"""
import json, sys, zipfile
from collections import Counter
import numpy as np
from PIL import Image
NAMES = ['aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil_tank', 'playground', 'ship', 'wind_turbine']
tr = json.load(open('data/course-data/train/instances.json'))['images']; val = json.load(open('data/course-data/val/images.json'))['images']
pred = {int(a): int(b) for a, b in (l.split(',') for l in zipfile.ZipFile(sys.argv[1]).read('classification.csv').decode().strip().splitlines()[1:])}
def th(root, ims): return np.stack([np.asarray(Image.open(f'{root}/' + im['file_name']).convert('L').resize((64, 64), Image.BILINEAR), dtype=np.float32) for im in ims])
def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s
F = {'as is': lambda a: a, 'lr': lambda a: a[..., :, ::-1], 'ud': lambda a: a[..., ::-1, :], '180': lambda a: a[..., ::-1, ::-1]}
N, S, W = 64, 39, 25
def rng(o): return (S, N) if o == 1 else (0, W) if o == -1 else (0, N)
rel = {k: {'copy': set(), 'near': set(), 'strip': set()} for k in ('as is', 'flipped')}
for w in sorted({im['width'] for im in val}):
    tg = [im for im in val if im['width'] == w]; pool = [im for im in tr if im['width'] == w]
    if not pool: continue
    X, P = th('data/course-data/val', tg), th('data/course-data/train', pool); Pn, sp = norm(P)
    for fname, f in F.items():
        key = 'as is' if fname == 'as is' else 'flipped'; Xf = np.ascontiguousarray(f(X))
        A, sa = norm(Xf); C = A @ Pn.T; C[sa < 4] = 0; C[:, sp < 4] = 0; m = C.max(1)
        rel[key]['copy'] |= {tg[i]['id'] for i in np.nonzero(m >= 0.9)[0]}
        rel[key]['near'] |= {tg[i]['id'] for i in np.nonzero((m >= 0.7) & (m < 0.9))[0]}
        if w != 512: continue
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                if not ox and not oy: continue
                (r0, r1), (c0, c1) = rng(oy), rng(ox); (q0, q1), (d0, d1) = rng(-oy), rng(-ox)
                a, sa2 = norm(Xf[:, r0:r1, c0:c1]); b, sb = norm(P[:, q0:q1, d0:d1])
                C2 = a @ b.T; C2[sa2 < 4] = 0; C2[:, sb < 4] = 0
                rel[key]['strip'] |= {tg[i]['id'] for i in np.nonzero(C2.max(1) >= 0.95)[0]}
seen = rel['as is']['copy'] | rel['as is']['near'] | rel['as is']['strip']
flipped = rel['flipped']['copy'] | rel['flipped']['near'] | rel['flipped']['strip']
only_flip = flipped - seen
print(f"test images: {len(val)}")
for k in ('as is', 'flipped'):
    print(f"  {k:8s}: whole copy {len(rel[k]['copy'])}, near-copy {len(rel[k]['near'])}, strip {len(rel[k]['strip'])}, any {len(rel[k]['copy'] | rel[k]['near'] | rel[k]['strip'])}")
print(f"  already repeated the right way round: {len(seen)}; only flipped (what the flip test adds): {len(only_flip)}; both: {len(seen & flipped)}; neither: {len(val) - len(seen | flipped)}")
print(f"  only-flipped split: whole copy {len(rel['flipped']['copy'] - seen)}, near-copy only {len((rel['flipped']['near'] - rel['flipped']['copy']) - seen)}, strip only {len(rel['flipped']['strip'] - rel['flipped']['copy'] - rel['flipped']['near'] - seen)}")
c = Counter(pred[i] for i in only_flip); t = Counter(pred.values())
print('  only-flipped by predicted class: ' + ', '.join(f'{NAMES[k]} {c[k]}/{t[k]}' for k in range(9)))
