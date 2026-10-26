"""Three more ways a test image can repeat a labelled one (Pearson on thumbnails, >= 0.9 counted as a match).
A. 312 px tile-grid neighbours after flipping the test image (lr / ud / 180), 512 px.
B. Whole image across sizes: 256 px test vs 512 px labelled and 512 px test vs 256 px labelled, as is or flipped.
C. Crop at the same scale: a 256 px image as any 256 px window of a 512 px image (16 px cells), as is or flipped,
   test inside labelled and labelled inside test.

Run from the repository root (needs data/course-data and the holdout results on the cluster):

    python tools/analysis/cross_size_and_flipped_strips.py
"""
import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
tr = json.load(open('data/course-data/train/instances.json'))['images']; val = json.load(open('data/course-data/val/images.json'))['images']
def th(root, ims, n): return np.stack([np.asarray(Image.open(f'{root}/' + im['file_name']).convert('L').resize((n, n), Image.BILINEAR), dtype=np.float32) for im in ims])
def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s
F = {'as is': lambda a: a, 'lr': lambda a: a[..., :, ::-1], 'ud': lambda a: a[..., ::-1, :], '180': lambda a: a[..., ::-1, ::-1]}
T5, L5 = [im for im in val if im['width'] == 512], [im for im in tr if im['width'] == 512]
T2, L2 = [im for im in val if im['width'] == 256], [im for im in tr if im['width'] == 256]
# A
N, S, W = 64, 39, 25
X, P = th('data/course-data/val', T5, N), th('data/course-data/train', L5, N)
def rng(o): return (S, N) if o == 1 else (0, W) if o == -1 else (0, N)
plain = set()
for fname, f in F.items():
    Xf = np.ascontiguousarray(f(X)); hit = set()
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            if not ox and not oy: continue
            (r0, r1), (c0, c1) = rng(oy), rng(ox); (q0, q1), (d0, d1) = rng(-oy), rng(-ox)
            a, sa = norm(Xf[:, r0:r1, c0:c1]); b, sb = norm(P[:, q0:q1, d0:d1])
            C = a @ b.T; C[sa < 4] = 0; C[:, sb < 4] = 0
            hit |= set(np.nonzero(C.max(1) >= 0.95)[0].tolist())
    if fname == 'as is': plain = hit
    print(f'A. test 512 px images sharing a 200 px strip with a labelled tile, test image {fname}: {len(hit)}' + ('' if fname == 'as is' else f' (new, not found as is: {len(hit - plain)})'), flush=True)
# B
for name, (t_root, tg, pool) in {'256 px test vs 512 px labelled': ('data/course-data/val', T2, L5), '512 px test vs 256 px labelled': ('data/course-data/val', T5, L2)}.items():
    Xt, Pt = th(t_root, tg, 64), th('data/course-data/train', pool, 64)
    b, sb = norm(Pt); found = set()
    for f in F.values():
        a, sa = norm(np.ascontiguousarray(f(Xt))); C = a @ b.T; C[sa < 4] = 0; C[:, sb < 4] = 0
        found |= set(np.nonzero(C.max(1) >= 0.9)[0].tolist())
    print(f'B. {name}, whole image (any flip): {len(found)} of {len(tg)}', flush=True)
# C
def crops(small_root, small, big_root, big, label):
    t = th(small_root, small, 16); g = th(big_root, big, 32)
    tmpl = np.concatenate([np.ascontiguousarray(f(t)) for f in F.values()]); A, sa = norm(tmpl); A[sa < 4] = 0
    best = np.full(len(tmpl), -1.0)
    for s in range(0, len(g), 200):
        win = sliding_window_view(g[s:s + 200], (16, 16), axis=(1, 2)).reshape(-1, 256)
        B, sb = norm(win); B[sb < 4] = 0
        best = np.maximum(best, (A @ B.T).max(1))
    per_image = best.reshape(len(F), len(small)).max(0)
    print(f'C. {label}: 256 px window match >= 0.9: {(per_image >= 0.9).sum()} of {len(small)} (>= 0.8: {(per_image >= 0.8).sum()})', flush=True)
crops('data/course-data/val', T2, 'data/course-data/train', L5, '256 px test images inside 512 px labelled images')
crops('data/course-data/train', L2, 'data/course-data/val', T5, '256 px labelled images inside 512 px test images')
crops('data/course-data/train', L2, 'data/course-data/train', [im for im in tr if im['width'] == 512][:2000], 'control: 256 px labelled inside 512 px labelled (first 2000)')
