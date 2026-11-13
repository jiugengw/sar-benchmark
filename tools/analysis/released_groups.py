"""Score a submission ZIP on copies vs new images, using the val / test labels released on 13 Nov 2026.

Each val / test image is compared with every training image of the same size (64 px thumbnails, Pearson r, under the
dataset's flips: none, left-right, up-down, both). Groups, as in tools/analysis/holdout_groups.py:
  copies (r >= 0.9, not flipped), flipped copies (r >= 0.9, flipped), near-copies (0.7-0.9), borderline (0.6-0.7),
  new (< 0.6). "new6" = mean AP on new images over the six classes with enough new images (aircraft, bridge, car,
  oil tank, playground, ship).

    ~/.venvs/sar-yolo/bin/python tools/analysis/released_groups.py submissions/phase2/phase2.zip test
    -> results/released_labels/<zip stem>_<split>.json   (groups cached in results/released_labels/groups_<split>.json)
"""
import contextlib, io, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / 'scripts'))
import score_released as R

DATA = ROOT / 'data' / 'course-data'
OUT = ROOT / 'results' / 'released_labels'
FLIPS = [lambda a: a, lambda a: a[..., :, ::-1], lambda a: a[..., ::-1, :], lambda a: a[..., ::-1, ::-1]]
SIX = ['aircraft', 'bridge', 'car', 'oil_tank', 'playground', 'ship']


def thumb(path, n=64):
    return np.asarray(Image.open(path).convert('L').resize((n, n), Image.BILINEAR), dtype=np.float32)


def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s


def groups(split):
    cache = OUT / f'groups_{split}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    tr = json.loads((DATA / 'train' / 'instances.json').read_text())['images']
    q = json.loads((R.LABELS / f'{split}_instances.json').read_text())['images']
    best, kind, ref = {}, {}, {}
    for w in sorted({im['width'] for im in q}):
        tg = [im for im in q if im['width'] == w]; pool = [im for im in tr if im['width'] == w]
        if not pool: continue
        P, sp = norm(np.stack([thumb(DATA / 'train' / im['file_name']) for im in pool]))
        X = np.stack([thumb(DATA / split / im['file_name']) for im in tg])
        for k, f in enumerate(FLIPS):
            A, sa = norm(np.ascontiguousarray(f(X))); C = A @ P.T; C[sa < 4] = 0; C[:, sp < 4] = 0
            j = C.argmax(1); m = C[np.arange(len(tg)), j]
            for i, im in enumerate(tg):
                if m[i] > best.get(im['id'], -2):
                    best[im['id']], kind[im['id']], ref[im['id']] = float(m[i]), k, pool[j[i]]['file_name']
    g = {}
    for im in q:
        i = im['id']; r = best.get(i, -1)
        g[str(i)] = {'r': round(r, 4), 'flip': ['none', 'lr', 'ud', 'both'][kind.get(i, 0)], 'train_match': ref.get(i),
                     'group': ('flipped copy' if kind.get(i, 0) else 'copy') if r >= 0.9 else
                              'near-copy 0.7-0.9' if r >= 0.7 else 'borderline 0.6-0.7' if r >= 0.6 else 'new <0.6'}
    OUT.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps(g, indent=0))
    return g


def main(zip_path, split):
    g = groups(split); cls, det = R.load_zip(zip_path)
    order = ['copy', 'flipped copy', 'near-copy 0.7-0.9', 'borderline 0.6-0.7', 'new <0.6']
    sets = {k: [int(i) for i, v in g.items() if v['group'] == k] for k in order}
    sets['all copies (r >= 0.9)'] = sets['copy'] + sets['flipped copy']
    out = {'all': R.score(cls, det, split)}
    for k, ids in sets.items():
        if ids: out[k] = R.score(cls, det, split, ids, per_class=(k == 'new <0.6'))
    pc = out['new <0.6'].pop('per_class'); out['new <0.6'].pop('confusion_rows_true_cols_pred')
    out['new <0.6']['per_class_AP'] = {c: v['AP'] for c, v in pc.items()}
    out['new <0.6']['per_class_images'] = {c: v['images'] for c, v in pc.items()}
    aps = [pc[c]['AP'] for c in SIX if pc[c]['AP'] is not None]
    out['new6'] = round(float(np.mean(aps)), 2)
    p = OUT / f'{Path(zip_path).stem}_{split}.json'; p.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"{k:24s} {v['images']:5d} img  Δm {v['delta_m']:7.2f}  F1 {v['macro_f1']:6.2f}  mAP {v['mAP']:6.2f}  AP50 {v['AP50']:6.2f}")
    print('new6', out['new6'], '| new images per class', out['new <0.6']['per_class_images'])
    print('->', p)


if __name__ == '__main__':
    main(*sys.argv[1:3])
