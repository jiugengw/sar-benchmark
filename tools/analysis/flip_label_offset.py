"""For clean-holdout images that are flipped copies of training images: do their labels, turned back, equal the
training image's labels? Per transform: share of labels with an exact partner, and the pixel offsets otherwise.

Run from the repository root (needs data/course-data and the holdout results on the cluster):

    python tools/analysis/flip_label_offset.py results/ft40-s1r-memorise-337666/one2many/instances_val_holdout.json
"""
import json, sys
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
sys.path.insert(0, '.')
from sarbench.twins import clean_image_ids
N = 64
F = {'lr': lambda a: a[..., :, ::-1], 'ud': lambda a: a[..., ::-1, :], 'r180': lambda a: a[..., ::-1, ::-1]}
def box(b, t, W, H):
    x, y, w, h = b
    if t in ('lr', 'r180'): x = W - x - w
    if t in ('ud', 'r180'): y = H - y - h
    return [x, y, w, h]
def iou(a, b):
    iw = max(0, min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])); ih = max(0, min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    i = iw * ih; return i / (a[2]*a[3] + b[2]*b[3] - i + 1e-9)
hold = json.load(open(sys.argv[1])); tr = json.load(open('data/course-data/train/instances.json'))
hf = {im['file_name'] for im in hold['images']}; clean = set(clean_image_ids(hold['images']))
by = defaultdict(list)
for a in tr['annotations']: by[a['image_id']].append(a)
def th(ims): return np.stack([np.asarray(Image.open('data/course-data/train/' + im['file_name']).convert('L').resize((N, N), Image.BILINEAR), dtype=np.float32) for im in ims])
def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s
stats = defaultdict(lambda: defaultdict(list)); counts = Counter(); examples = []
for w in (512, 256):
    pool = [im for im in tr['images'] if im['width'] == w and im['file_name'] not in hf]
    P, sp = norm(th(pool)); tg = [im for im in hold['images'] if im['width'] == w and im['id'] in clean]; X = th(tg)
    done = set()
    for t, f in F.items():
        A, sa = norm(np.ascontiguousarray(f(X))); C = A @ P.T; C[sa < 4] = 0; C[:, sp < 4] = 0
        for i in np.nonzero(C.max(1) >= 0.9)[0]:
            if tg[i]['id'] in done: continue
            done.add(tg[i]['id']); src = pool[int(C[i].argmax())]
            mine = [a for a in by[tg[i]['id']]]; theirs = by[src['id']]
            counts[(w, t, 'labels here', len(mine) == len(theirs))] += 1
            for a in mine:
                b = box(a['bbox'], t, w, w)
                cands = [(iou(b, s['bbox']), s) for s in theirs if s['category_id'] == a['category_id']]
                if not cands: stats[(w, t)]['no partner'].append(1); continue
                v, s = max(cands, key=lambda z: z[0])
                stats[(w, t)]['iou'].append(v)
                stats[(w, t)]['dx'].append(s['bbox'][0] - b[0]); stats[(w, t)]['dy'].append(s['bbox'][1] - b[1])
                stats[(w, t)]['dw'].append(s['bbox'][2] - b[2]); stats[(w, t)]['dh'].append(s['bbox'][3] - b[3])
                if len(examples) < 6 and v < 0.99: examples.append((w, t, [round(z, 1) for z in a['bbox']], [round(z, 1) for z in b], [round(z, 1) for z in s['bbox']]))
for k, d in sorted(stats.items()):
    io = np.array(d['iou'])
    print(f'{k[0]} px {k[1]:4s}: {len(io)} labels with a partner, {len(d["no partner"])} without; IoU with the training label: '
          f'exact (>=0.99) {100*(io>=0.99).mean():.0f}%, median {np.median(io):.3f}; offsets px median dx {np.median(d["dx"]):+.2f} '
          f'dy {np.median(d["dy"]):+.2f} dw {np.median(d["dw"]):+.2f} dh {np.median(d["dh"]):+.2f}; |dx| mean {np.mean(np.abs(d["dx"])):.2f}')
print('same number of labels in the pair:', dict(counts))
print('examples (size, transform, holdout label, turned back, training label):'); [print(' ', e) for e in examples]
