# Why do flipped copies score ~91 when the model trains on those exact flipped pixels, while the same scenes as-is
# score ~98? (29 Oct; analysis only, no label is copied anywhere.) For the clean holdout images that are
# flipped copies of training images (64 px correlation >= 0.9 under a flip), with the holdout twin of refit 2 (340514):
#   A  the holdout flipped copies, the model's own boxes, on their own labels (what the test sees)
#   B  their training originals as they are, on the originals' labels (trained-on, as-is)
#   C  the training originals flipped here (pixels mirrored exactly), on the originals' labels flipped with the
#      dataset's convention (x' = W - x - w - 1 at 512 px, -2 at 256) (trained-on, flipped: dataset flips show exactly this)
#   D  B's boxes flipped with the convention, scored on the flipped copies' own labels (perfect recall of the original)
# plus, for matched boxes, the median signed edge error (prediction - label, px at 512) along the flipped axis: if the
# model still mirrors boxes exactly (Ultralytics' flips in the first ~260 epochs), A and C sit about +1 px off.
#   python flip_gap.py WEIGHTS HOLDOUT_RESULTS_DIR OUT_DIR
import contextlib, io, json, subprocess, sys, tempfile, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo')
from sarbench.metrics import detection_metrics
from sarbench.twins import clean_image_ids
from hedge_study import CHOSEN_TIERS, hedge_tiers
from round_boxes import round_predictions

weights, R, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
TR = Path('data/course-data/train')
hg = json.load(open(R / 'instances_val_holdout.json')); hims = {im['id']: im for im in hg['images']}
train = json.load(open(TR / 'instances.json')); tims = {im['id']: im for im in train['images']}
names = {c['id']: c['name'] for c in train['categories']}
clean = set(clean_image_ids(hg['images'])); hf = {im['file_name'] for im in hg['images']}
ann_t = defaultdict(list)
for a in train['annotations']:
    ann_t[a['image_id']].append(a)
ann_h = defaultdict(list)
for a in hg['annotations']:
    ann_h[a['image_id']].append(a)


def load(root, im, n=None):
    g = Image.open(root / im['file_name']).convert('L')
    return np.asarray(g.resize((n, n), Image.BILINEAR) if n else g, dtype=np.float32)


def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s


FL = {1: lambda a: a[..., :, ::-1], 2: lambda a: a[..., ::-1, :], 3: lambda a: a[..., ::-1, ::-1]}
pairs = {}  # holdout id -> (training id, flip kind, pixel-exact?)
for w in sorted({im['width'] for im in hg['images']}):
    tg = [im for im in hg['images'] if im['width'] == w and im['id'] in clean]
    pool = [im for im in train['images'] if im['width'] == w and im['file_name'] not in hf]
    if not tg or not pool:
        continue
    P, sp = norm(np.stack([load(TR, im, 64) for im in pool])); X = np.stack([load(TR, im, 64) for im in tg])
    A0, sa = norm(X); same = (A0 @ P.T).max(1)
    for k, f in FL.items():
        A, sa = norm(np.ascontiguousarray(f(X))); C = A @ P.T; C[sa < 4] = 0; C[:, sp < 4] = 0
        j, m = C.argmax(1), C.max(1)
        for i, im in enumerate(tg):
            if m[i] >= 0.9 and m[i] > same[i] and m[i] > pairs.get(im['id'], (0, 0, 0, -1))[3]:
                pairs[im['id']] = (pool[j[i]]['id'], k, False, float(m[i]))
for h, (t, k, _, r) in list(pairs.items()):
    a, b = load(TR, hims[h]), FL[k](load(TR, tims[t]))
    exact = a.shape == b.shape and np.corrcoef(a.ravel(), b.ravel())[0, 1] >= 0.999
    pairs[h] = (t, k, bool(exact), r)
print(f'{len(pairs)} clean holdout flipped copies; pixel-exact {sum(p[2] for p in pairs.values())}; '
      f'by flip lr/ud/180: {[sum(1 for p in pairs.values() if p[1] == k) for k in (1, 2, 3)]}', flush=True)


def flip_box(b, k, W, H, off):
    x, y, w, h = b
    if k in (1, 3):
        x = W - x - w - off
    if k in (2, 3):
        y = H - y - h - off
    return [x, y, w, h]


def offset(W):
    return 2 if W <= 300 else 1


def predict(ims, root, tag):
    cache = out / f'{tag}_predictions.json'
    if cache.exists():
        return json.loads(cache.read_text())
    with tempfile.TemporaryDirectory(prefix='sar-flipgap-') as tmp:
        tmp = Path(tmp); view = tmp / 'data'; (view / 'train').mkdir(parents=True); (view / 'val').mkdir()
        (view / 'train/labels.csv').symlink_to((TR / 'labels.csv').resolve())
        for im in ims:
            target = view / 'val' / im['file_name']; target.parent.mkdir(parents=True, exist_ok=True)
            src = (root / im['file_name']).resolve()
            target.symlink_to(src)
        (view / 'val/images.json').write_text(json.dumps({'images': ims}))
        subprocess.run([sys.executable, '-u', 'models/yolo/make_submission.py', '--weights', weights, '--detection-branch',
                        'one2many', '--data', str(view), '--out', str(tmp / 'p.zip'), '--batch', '16'],
                       check=True, stdout=subprocess.DEVNULL)
        preds = json.loads(zipfile.ZipFile(tmp / 'p.zip').read('detection.json'))
    cache.write_text(json.dumps(preds))
    return preds


origs = sorted({t for t, *_ in pairs.values()})
B_pred = predict([tims[t] for t in origs], TR, 'originals')
# C: the originals mirrored exactly, saved losslessly
flipdir = out / 'flipped_originals'
flip_ims = []
for h, (t, k, *_) in pairs.items():
    im = tims[t]; name = f'images/flip{k}_h{h}.png'  # per holdout copy: two copies can share one original
    (flipdir / 'images').mkdir(parents=True, exist_ok=True)
    if not (flipdir / name).exists():
        Image.fromarray(np.ascontiguousarray(FL[k](np.asarray(Image.open(TR / im['file_name']).convert('L'))))).save(flipdir / name)
    flip_ims.append({'id': 10_000_000 + h, 'file_name': name, 'width': im['width'], 'height': im['height']})
C_pred = predict(flip_ims, flipdir, 'flipped_originals')


def gt_file(path, images, anns):
    path.write_text(json.dumps({'images': images, 'categories': train['categories'], 'annotations': anns}))
    return str(path)


gB = gt_file(out / 'gt_B.json', [tims[t] for t in origs], [a for t in origs for a in ann_t[t]])
cann, n = [], 0
for h, (t, k, *_) in pairs.items():
    im = tims[t]
    for a in ann_t[t]:
        n += 1
        cann.append(dict(a, id=n, image_id=10_000_000 + h, bbox=flip_box(a['bbox'], k, im['width'], im['height'], offset(im['width']))))
gC = gt_file(out / 'gt_C.json', flip_ims, cann)
D_pred = []
for h, (t, k, *_) in pairs.items():
    im = tims[t]
    D_pred += [dict(p, image_id=h, bbox=flip_box(p['bbox'], k, im['width'], im['height'], offset(im['width']))) for p in B_pred if p['image_id'] == t]
A_pred = [p for p in json.load(open(R / 'predictions_val.json')) if p['image_id'] in pairs]
widths = {**{i: im['width'] for i, im in hims.items()}, **{i: im['width'] for i, im in tims.items()}, **{im['id']: im['width'] for im in flip_ims}}


def score(gt, preds, ids):
    with contextlib.redirect_stdout(io.StringIO()):
        raw = detection_metrics(gt, preds, ids)
        sub = detection_metrics(gt, hedge_tiers(round_predictions(preds, widths, 'whole-pixel-classes'), CHOSEN_TIERS), ids)
    return raw, sub


def row(label, r):
    raw, sub = r
    print(f"{label:58s} mAP {100 * sub['mAP']:5.1f} (raw {100 * raw['mAP']:5.1f}, AP50 {100 * sub['AP50']:5.1f}) | " +
          ', '.join(f'{k[:6]} {100 * v:5.1f}' for k, v in sub['per_class_AP'].items() if v >= 0), flush=True)


for subset, keep in (('all', lambda p: True), ('pixel-exact', lambda p: p[2]), ('re-processed', lambda p: not p[2])):
    hs = sorted(h for h, p in pairs.items() if keep(p))
    if not hs:
        continue
    ts = sorted({pairs[h][0] for h in hs})
    print(f'\n== {subset}: {len(hs)} flipped copies, {sum(len(ann_h[h]) for h in hs)} labels (rounded + hedged as submitted)')
    row('A flipped copies, own boxes, own labels (the test case)', score(str(R / 'instances_val_holdout.json'), A_pred, hs))
    row('B training originals as-is (trained on)', score(gB, B_pred, ts))
    row('C originals flipped here, convention labels (trained on)', score(gC, C_pred, [10_000_000 + h for h in hs]))
    row('D originals\' boxes flipped onto the copies\' labels', score(str(R / 'instances_val_holdout.json'), D_pred, hs))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1); return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def edges(preds, anns_by_img, img_w, kind_of):
    """Median signed (prediction - label) of the left/top and right/bottom edges along each image's flipped axis,
    in px at 512, for labels matched by a box scoring >= 0.3 at IoU >= 0.5; per class."""
    by = defaultdict(list)
    for p in preds:
        if p['score'] >= 0.3:
            by[p['image_id']].append(p)
    d = defaultdict(list)
    for i, anns in anns_by_img.items():
        k, s = kind_of(i), 512 / img_w(i)
        for a in anns:
            cands = [(iou(a['bbox'], p['bbox']), p) for p in by[i] if p['category_id'] == a['category_id']]
            cands = [c for c in cands if c[0] >= 0.5]
            if not cands:
                continue
            p = max(cands, key=lambda c: c[0])[1]
            for axis in ((0,) if k == 1 else (1,) if k == 2 else (0, 1)) if k else (0, 1):
                lo = s * (p['bbox'][axis] - a['bbox'][axis])
                hi = s * ((p['bbox'][axis] + p['bbox'][axis + 2]) - (a['bbox'][axis] + a['bbox'][axis + 2]))
                d[names[a['category_id']]].append((lo, hi))
    return {k: (len(v), np.median([x[0] for x in v]), np.median([x[1] for x in v])) for k, v in d.items()}


print('\n== median signed edge error along the flipped axis, prediction - label, px at 512 (n matched boxes)')
eA = edges(A_pred, {h: ann_h[h] for h in pairs}, lambda i: hims[i]['width'], lambda i: pairs[i][1])
eC = edges(C_pred, defaultdict(list, {10_000_000 + h: [a for a in cann if a['image_id'] == 10_000_000 + h] for h in pairs}),
           lambda i: hims[i - 10_000_000]['width'], lambda i: pairs[i - 10_000_000][1])
eB = edges(B_pred, {t: ann_t[t] for t in origs}, lambda i: tims[i]['width'], lambda i: 0)
for k in sorted(set(eA) | set(eB) | set(eC)):
    fmt = lambda e: f'n{e[0]:4d} lo {e[1]:+5.2f} hi {e[2]:+5.2f}' if e else f'{"-":>22s}'
    print(f'  {k:13s} A copies {fmt(eA.get(k))} | C flipped originals {fmt(eC.get(k))} | B as-is (both axes) {fmt(eB.get(k))}')
print('FLIP_GAP_DONE')
