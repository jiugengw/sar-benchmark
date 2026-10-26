"""Are the 'crop' matches real? Labelled 256 px images matched as a 256 px window of a labelled 512 px image (16 px
cells, any flip, r >= 0.9): refine the offset at 4 px cells, then check whether the 256 image's boxes have a same-class
box of the 512 image at the same place (centre within 6 px). Real crops agree; chance matches don't.

Run from the repository root (needs data/course-data and the holdout results on the cluster):

    python tools/analysis/crop_match_check.py
"""
import json
from collections import Counter, defaultdict
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
d = json.load(open('data/course-data/train/instances.json')); tr = d['images']
by = defaultdict(list)
for a in d['annotations']: by[a['image_id']].append(a)
def th(ims, n): return np.stack([np.asarray(Image.open('data/course-data/train/' + im['file_name']).convert('L').resize((n, n), Image.BILINEAR), dtype=np.float32) for im in ims])
def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s
F = [lambda a: a, lambda a: a[..., :, ::-1], lambda a: a[..., ::-1, :], lambda a: a[..., ::-1, ::-1]]
small = [im for im in tr if im['width'] == 256]; big = [im for im in tr if im['width'] == 512][:2000]
t = th(small, 16); g = th(big, 32)
tm = np.concatenate([np.ascontiguousarray(f(t)) for f in F]); A, sa = norm(tm); A[sa < 4] = 0
best = np.full(len(tm), -1.0); where = np.zeros((len(tm), 3), int)
for s in range(0, len(g), 200):
    win = sliding_window_view(g[s:s + 200], (16, 16), axis=(1, 2)); n = win.shape[0]
    B, sb = norm(win.reshape(-1, 256)); B[sb < 4] = 0
    C = A @ B.T; m = C.max(1); k = C.argmax(1); better = m > best
    best[better] = m[better]; idx = np.stack([k // 289 + s, (k % 289) // 17, k % 17], 1); where[better] = idx[better]
agree, classes = [], Counter()
for j in np.nonzero(best >= 0.9)[0]:
    fi, i = divmod(j, len(small)); sm, bg = small[i], big[where[j, 0]]
    oy, ox = where[j, 1] * 16, where[j, 2] * 16
    mine = by[sm['id']]
    if not mine: continue
    def flipc(cx, cy):
        if fi in (1, 3): cx = 256 - cx
        if fi in (2, 3): cy = 256 - cy
        return cx, cy
    theirs = [(a['category_id'], a['bbox'][0] + a['bbox'][2] / 2 - ox, a['bbox'][1] + a['bbox'][3] / 2 - oy) for a in by[bg['id']]]
    hits = []
    for a in mine:
        cx, cy = flipc(a['bbox'][0] + a['bbox'][2] / 2, a['bbox'][1] + a['bbox'][3] / 2)
        hits.append(any(c == a['category_id'] and abs(x - cx) <= 6 + 16 and abs(y - cy) <= 6 + 16 for c, x, y in theirs))
    agree.append(np.mean(hits)); classes[Counter(a['category_id'] for a in mine).most_common(1)[0][0]] += 1
print(f'labelled 256 px images matched inside labelled 512 px images (r >= 0.9): {len(agree)} with boxes; '
      f'share of their boxes with a same-class box nearby in the 512 image (within the 16 px window step): {100*np.mean(agree):.0f}%')
print('their main class ids:', dict(classes.most_common()))
