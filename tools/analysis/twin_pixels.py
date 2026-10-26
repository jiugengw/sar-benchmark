"""Full-resolution pixel correlation of holdout images with their best whole-image training match: identity twins vs
flipped copies (flipped back first). Re-saved JPEGs give ~0.99+; a re-captured scene (new speckle) much less.

Run from the repository root (needs data/course-data and the holdout results on the cluster):

    python tools/analysis/twin_pixels.py results/ft40-s1r-memorise-337666/one2many/instances_val_holdout.json
"""
import json, sys
import numpy as np
from PIL import Image
sys.path.insert(0, '.')
from sarbench.twins import clean_image_ids
gt = json.load(open(sys.argv[1])); tr = json.load(open('data/course-data/train/instances.json'))['images']
hf = {im['file_name'] for im in gt['images']}; clean = set(clean_image_ids(gt['images']))
def load(im, n=None):
    x = Image.open('data/course-data/train/' + im['file_name']).convert('L')
    return np.asarray(x.resize((n, n), Image.BILINEAR) if n else x, dtype=np.float32)
def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s
F = [lambda a: a, lambda a: a[..., :, ::-1], lambda a: a[..., ::-1, :], lambda a: a[..., ::-1, ::-1]]
out = {'twins (as is)': [], 'flipped copies': []}
for w in (512, 256):
    tg = [im for im in gt['images'] if im['width'] == w]; pool = [im for im in tr if im['width'] == w and im['file_name'] not in hf]
    P, sp = norm(np.stack([load(im, 64) for im in pool])); X = np.stack([load(im, 64) for im in tg])
    for k, f in enumerate(F):
        A, sa = norm(np.ascontiguousarray(f(X))); C = A @ P.T; C[sa < 4] = 0; C[:, sp < 4] = 0
        for i in np.nonzero(C.max(1) >= 0.9)[0]:
            im = tg[i]
            if (k == 0) != (im['id'] not in clean): continue  # as-is twins are the non-clean images; flipped copies are clean
            a = f(load(im)[None])[0].ravel(); b = load(pool[int(C[i].argmax())]).ravel()
            out['twins (as is)' if k == 0 else 'flipped copies'].append(np.corrcoef(a, b)[0, 1])
for k, v in out.items():
    v = np.array(v)
    print(f'{k}: {len(v)} pairs, full-resolution pixel correlation median {np.median(v):.3f}; >=0.99: {100*np.mean(v>=0.99):.0f}%, 0.9-0.99: {100*np.mean((v>=0.9)&(v<0.99)):.0f}%, <0.9: {100*np.mean(v<0.9):.0f}%')
