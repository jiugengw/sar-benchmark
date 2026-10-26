"""Several runs on the same holdout groups (rounded + hedged): official-mix (clean), then mAP on identity twins
(re-captured scenes), flipped copies, near-copies (r 0.7-0.9) and new scenes (r < 0.6).

Run from the repository root (needs data/course-data and the holdout results on the cluster):

    python tools/analysis/holdout_groups.py ft40-s1r-memorise-337666 ft40-s1r-flipmem-338727   # results/<run>/one2many
"""
import contextlib, io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo')
from official_mix_score import NAMES, official_shares
from resolution_probe import coco
from sarbench.twins import clean_image_ids
from sarbench.metrics import detection_metrics
from hedge_study import CHOSEN_TIERS, hedge_tiers
from round_boxes import round_predictions
runs = sys.argv[1:]
gt_path = f'results/{runs[0]}/one2many/instances_val_holdout.json'
gt = json.load(open(gt_path)); tr = json.load(open('data/course-data/train/instances.json'))['images']
hf = {im['file_name'] for im in gt['images']}; clean = set(clean_image_ids(gt['images'])); images = {im['id']: im for im in gt['images']}
def load(im, n): return np.asarray(Image.open('data/course-data/train/' + im['file_name']).convert('L').resize((n, n), Image.BILINEAR), dtype=np.float32)
def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s
F = [lambda a: a, lambda a: a[..., :, ::-1], lambda a: a[..., ::-1, :], lambda a: a[..., ::-1, ::-1]]
best, kind = {}, {}
for w in sorted({im['width'] for im in gt['images']}):
    tg = [im for im in gt['images'] if im['width'] == w]; pool = [im for im in tr if im['width'] == w and im['file_name'] not in hf]
    if not tg or not pool: continue
    P, sp = norm(np.stack([load(im, 64) for im in pool])); X = np.stack([load(im, 64) for im in tg])
    for k, f in enumerate(F):
        A, sa = norm(np.ascontiguousarray(f(X))); C = A @ P.T; C[sa < 4] = 0; C[:, sp < 4] = 0; m = C.max(1)
        for i, im in enumerate(tg):
            if m[i] > best.get(im['id'], -2): best[im['id']], kind[im['id']] = float(m[i]), k
groups = {'twins (re-captured)': [i for i in images if i not in clean],
          'flipped copies': [i for i in clean if best.get(i, -1) >= 0.9 and kind.get(i, 0) > 0],
          'near-copies 0.7-0.9': [i for i in clean if 0.7 <= best.get(i, -1) < 0.9],
          'new <0.6': [i for i in clean if best.get(i, -1) < 0.6]}
shares = official_shares(Path('submissions/memorise-lr2-332326/memorise-lr2-332326.zip'), Path('data/course-data/val/images.json'))
print(f"{'run':30s} {'off-mix':>7s} " + ' '.join(f'{g[:19]:>19s}' for g in groups))
print(f"{'(images)':30s} {'':7s} " + ' '.join(f'{len(v):19d}' for v in groups.values()))
for run in runs:
    b = hedge_tiers(round_predictions(json.load(open(f'results/{run}/one2many/predictions_val.json')), {i: im['width'] for i, im in images.items()}, 'whole-pixel-classes'), CHOSEN_TIERS)
    small = {i for i in clean if images[i]['width'] == 256}
    with contextlib.redirect_stdout(io.StringIO()):
        _, x = coco(gt_path, b, small); _, y = coco(gt_path, b, clean - small)
    mix = np.mean([100 * (y[n] if x.get(n, float('nan')) != x.get(n, float('nan')) else x[n] if y.get(n, float('nan')) != y.get(n, float('nan')) else shares[n] * x[n] + (1 - shares[n]) * y[n]) for n in NAMES])
    print(f'{run[:30]:30s} {mix:7.2f} ' + ' '.join(f"{100 * detection_metrics(gt_path, b, sorted(ids))['mAP']:19.1f}" for ids in groups.values()), flush=True)
