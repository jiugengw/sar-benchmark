"""Flip-memorise test vs its control on the clean holdout (rounded + two-tier hedged, as submitted): official-mix and
per class, then mAP on the clean images that are flipped copies of training images vs the other clean images.
Last line: FLIPDECISION <official-mix gain> <flipped-copy gain> <rest gain>.

Run from the repository root (needs data/course-data and the holdout results on the cluster):

    python tools/analysis/flip_eval.py ft40-s1r-memorise-337666 ft40-s1r-flipmem-338727
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
control, test = sys.argv[1], sys.argv[2]
shares = official_shares(Path('submissions/memorise-lr2-332326/memorise-lr2-332326.zip'), Path('data/course-data/val/images.json'))
gt_path = Path(f'results/{control}/one2many/instances_val_holdout.json')
gt = json.loads(gt_path.read_text()); images = {im['id']: im for im in gt['images']}; clean = set(clean_image_ids(gt['images']))
widths = {i: im['width'] for i, im in images.items()}
# clean holdout images that are left-right / up-down / 180-degree copies of training images (64 px thumbnails, r >= 0.9)
tr = json.load(open('data/course-data/train/instances.json')); hf = {im['file_name'] for im in gt['images']}
def th(ims): return np.stack([np.asarray(Image.open('data/course-data/train/' + im['file_name']).convert('L').resize((64, 64), Image.BILINEAR), dtype=np.float32) for im in ims])
def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s
flipped = set()
for w in sorted({im['width'] for im in gt['images']}):
    pool = [im for im in tr['images'] if im['width'] == w and im['file_name'] not in hf]
    tg = [im for im in gt['images'] if im['width'] == w and im['id'] in clean]
    if not pool or not tg: continue
    P, sp = norm(th(pool)); X = th(tg)
    for f in (lambda a: a[:, :, ::-1], lambda a: a[:, ::-1], lambda a: a[:, ::-1, ::-1]):
        A, sa = norm(np.ascontiguousarray(f(X))); C = A @ P.T; C[sa < 4] = 0; C[:, sp < 4] = 0
        flipped |= {tg[i]['id'] for i in np.nonzero(C.max(1) >= 0.9)[0]}
rest = clean - flipped
print(f'clean holdout: {len(clean)} images, {len(flipped)} flipped copies of training images, {len(rest)} others')
rows, sub = {}, {}
for run in (control, test):
    r = Path(f'results/{run}/one2many')
    b = hedge_tiers(round_predictions(json.loads((r / 'predictions_val.json').read_text()), widths, 'whole-pixel-classes'), CHOSEN_TIERS)
    small = {i for i in clean if images[i]['width'] == 256}
    with contextlib.redirect_stdout(io.StringIO()):
        _, x = coco(gt_path, b, small); _, y = coco(gt_path, b, clean - small)
    rows[run] = {n: 100 * (y[n] if x.get(n, float('nan')) != x.get(n, float('nan')) else x[n] if y.get(n, float('nan')) != y.get(n, float('nan')) else shares[n] * x[n] + (1 - shares[n]) * y[n]) for n in NAMES}
    sub[run] = {k: detection_metrics(str(gt_path), b, sorted(ids)) for k, ids in (('flipped', flipped), ('rest', rest))}
print(f"{'class':13s} {control[:24]:>24s} {test[:24]:>24s}  change | flipped-copy images: control -> test | other clean images")
for n in NAMES:
    a, t = rows[control][n], rows[test][n]
    fa, ft = (100 * sub[k]['flipped']['per_class_AP'].get(n, -1) for k in (control, test))
    ra, rt = (100 * sub[k]['rest']['per_class_AP'].get(n, -1) for k in (control, test))
    print(f'{n:13s} {a:24.1f} {t:24.1f} {t - a:+7.1f} | {fa:6.1f} -> {ft:6.1f} | {ra:6.1f} -> {rt:6.1f}')
m = {k: np.mean(list(rows[k].values())) for k in rows}
fm = {k: 100 * sub[k]['flipped']['mAP'] for k in rows}; rm = {k: 100 * sub[k]['rest']['mAP'] for k in rows}
print(f"{'official-mix':13s} {m[control]:24.2f} {m[test]:24.2f} {m[test] - m[control]:+7.2f} | {fm[control]:6.1f} -> {fm[test]:6.1f} | {rm[control]:6.1f} -> {rm[test]:6.1f}")
print(f'FLIPDECISION {m[test] - m[control]:.2f} {fm[test] - fm[control]:.2f} {rm[test] - rm[control]:.2f}')
