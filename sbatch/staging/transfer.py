# Label transfer from the most similar labelled image (30 Oct; EXPERIMENT pending the TA's approval, see
# no-twin-label-copying). For each target image: find the most similar labelled image (64 px correlation, as-is and in
# the three flips), and if the similarity falls in a chosen range, copy that image's labels of the chosen classes
# (flipped with the dataset's convention) into the predictions at score 1.0, so they rank above every model box.
#   holdout study: python transfer.py study                 (340514's holdout predictions, rounded + hedged + tail)
#   test ZIP:      python transfer.py zip IN.zip OUT.zip LO HI CLASSES MODE
import contextlib, io, json, sys, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo')
from round_boxes import round_predictions

TR = Path('data/course-data/train')
train = json.load(open(TR / 'instances.json'))
names = {c['id']: c['name'] for c in train['categories']}; ids_of = {v: k for k, v in names.items()}
ann = defaultdict(list)
for a in train['annotations']:
    ann[a['image_id']].append(a)
FL = [lambda a: a, lambda a: a[..., :, ::-1], lambda a: a[..., ::-1, :], lambda a: a[..., ::-1, ::-1]]


def thumb(root, im):
    return np.asarray(Image.open(Path(root) / im['file_name']).convert('L').resize((64, 64), Image.BILINEAR), dtype=np.float32)


def norm(x):
    x = x.reshape(len(x), -1); s = x.std(1); x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6), s


def match(targets, troot, pool):
    """target id -> (similarity, flip kind, labelled image id) of the best match of the same width."""
    best = {}
    for w in sorted({im['width'] for im in targets}):
        tg = [im for im in targets if im['width'] == w]; pl = [im for im in pool if im['width'] == w]
        if not tg or not pl:
            continue
        P, sp = norm(np.stack([thumb(TR, im) for im in pl])); X = np.stack([thumb(troot, im) for im in tg])
        for k, f in enumerate(FL):
            A, sa = norm(np.ascontiguousarray(f(X))); C = A @ P.T; C[sa < 4] = 0; C[:, sp < 4] = 0
            j, m = C.argmax(1), C.max(1)
            for i, im in enumerate(tg):
                if m[i] > best.get(im['id'], (-2,))[0]:
                    best[im['id']] = (float(m[i]), k, pl[j[i]]['id'])
    return best


def flip(b, k, W, H):
    x, y, w, h = b; off = 2 if W <= 300 else 1
    if k in (1, 3): x = W - x - w - off
    if k in (2, 3): y = H - y - h - off
    return [x, y, w, h]


def transfer(preds, best, images, lo, hi, classes, mode):
    """Copy the matched image's labels of `classes` into images whose match similarity is in [lo, hi). mode 'add' keeps
    the model's boxes (ranked below); 'replace' drops the model's boxes of those classes in those images."""
    cids = {ids_of[c] for c in classes}
    hit = {i: v for i, v in best.items() if lo <= v[0] < hi}
    out = [p for p in preds if not (mode == 'replace' and p['image_id'] in hit and p['category_id'] in cids)]
    added = []
    for i, (r, k, t) in hit.items():
        im = images[i]
        added += [{'image_id': i, 'category_id': a['category_id'], 'score': 1.0,
                   'bbox': flip(a['bbox'], k, im['width'], im['height'])} for a in ann[t] if a['category_id'] in cids]
    keep, n = [], defaultdict(int)  # the scorer keeps 100 boxes per image and class: the copied ones first
    for p in sorted(added + out, key=lambda p: -p['score']):
        key = p['image_id'], p['category_id']
        if n[key] < 100:
            n[key] += 1; keep.append(p)
    return keep, len(hit), len(added)


STATIC = ['airport', 'wind_turbine', 'playground', 'bridge']


def study():
    from hedge_study import CHOSEN_TIERS, hedge_tiers
    from official_mix_score import NAMES, official_shares
    from resolution_probe import coco
    from sarbench.metrics import detection_metrics
    from sarbench.twins import clean_image_ids
    R = Path('results/ft40-s1r-fliprefit-340514/one2many'); gt_path = str(R / 'instances_val_holdout.json')
    gt = json.load(open(gt_path)); images = {im['id']: im for im in gt['images']}; widths = {i: im['width'] for i, im in images.items()}
    clean = set(clean_image_ids(gt['images'])); twins = set(images) - clean; hf = {im['file_name'] for im in gt['images']}
    best = match(gt['images'], TR, [im for im in train['images'] if im['file_name'] not in hf])
    base = hedge_tiers(round_predictions(json.load(open(R / 'predictions_val.json')), widths, 'whole-pixel-classes'), CHOSEN_TIERS)
    low = [p for p in json.load(open('results/tail-lowconf-340514/predictions_lowconf.json')) if p['score'] < 0.001]
    floor = min(p['score'] for p in base); room = defaultdict(lambda: 100)
    for p in base:
        room[p['image_id'], p['category_id']] -= 1
    for p in sorted(round_predictions(low, widths, 'whole-pixel-classes'), key=lambda p: -p['score']):
        if room[p['image_id'], p['category_id']] > 0:
            room[p['image_id'], p['category_id']] -= 1; base.append({**p, 'score': p['score'] * floor / 0.001 * 0.999})
    shares = official_shares(Path('submissions/memorise-lr2-332326/memorise-lr2-332326.zip'), Path('data/course-data/val/images.json'))

    def mix(boxes, ids):
        small = {i for i in ids if images[i]['width'] == 256}; keep = [b for b in boxes if b['image_id'] in ids]
        with contextlib.redirect_stdout(io.StringIO()):
            _, x = coco(gt_path, keep, small); _, y = coco(gt_path, keep, ids - small)
        v = []
        for n in NAMES:
            s, l = x.get(n, float('nan')), y.get(n, float('nan'))
            v.append(l if s != s else s if l != l else shares[n] * s + (1 - shares[n]) * l)
        return 100 * float(np.nanmean(v))

    def grp(lo, hi, pool):
        return {i for i in pool if lo <= best.get(i, (-1,))[0] < hi}
    groups = {'near 0.7-0.9': grp(0.7, 0.9, clean), 'mid 0.6-0.7': grp(0.6, 0.7, clean)}

    def mapof(boxes, ids):
        with contextlib.redirect_stdout(io.StringIO()):
            return 100 * detection_metrics(gt_path, boxes, sorted(ids))['mAP']
    print(f"holdout: {len(clean)} clean images; near 0.7-0.9 {len(groups['near 0.7-0.9'])}, mid 0.6-0.7 {len(groups['mid 0.6-0.7'])}, twins {len(twins)}")
    print(f"{'variant':52s} {'off-mix':>7s} {'near':>6s} {'mid':>6s} {'twins':>6s}  images / boxes copied")
    print(f"{'current pipeline (no transfer)':52s} {mix(base, clean):7.2f} {mapof(base, groups['near 0.7-0.9']):6.1f} {mapof(base, groups['mid 0.6-0.7']):6.1f} {mapof(base, twins):6.1f}")
    for lo, hi in ((0.7, 0.9), (0.6, 0.9), (0.8, 0.9), (0.6, 1.01)):
        for classes in (STATIC, STATIC + ['oil_tank'], ['airport', 'wind_turbine', 'playground']):
            for mode in ('add', 'replace'):
                b, nimg, nbox = transfer(base, best, images, lo, hi, classes, mode)
                label = f"{lo}-{min(hi, 1.0)} {'+'.join(c[:4] for c in classes)} {mode}"
                print(f"{label:52s} {mix(b, clean):7.2f} {mapof(b, groups['near 0.7-0.9']):6.1f} {mapof(b, groups['mid 0.6-0.7']):6.1f} {mapof(b, twins):6.1f}  {nimg} / {nbox}", flush=True)
    print('TRANSFER_STUDY_DONE')


def make_zip(src, dst, lo, hi, classes, mode):
    VA = Path('data/course-data/val'); val = json.load(open(VA / 'images.json'))['images']; images = {im['id']: im for im in val}
    best = match(val, VA, train['images'])  # every labelled image (train + holdout) can be the match
    z = zipfile.ZipFile(src); members = {n: z.read(n) for n in z.namelist()}
    preds = json.loads(members['detection.json'])
    out, nimg, nbox = transfer(preds, best, images, lo, hi, classes, mode)
    members['detection.json'] = json.dumps(out).encode()
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as o:
        for n, d in members.items():
            o.writestr(n, d)
    print(f'{dst}: copied labels into {nimg} test images ({nbox} boxes; classes {classes}, similarity {lo}-{hi}, {mode}); '
          f'{len(preds)} -> {len(out)} boxes; classification.csv unchanged')


if __name__ == '__main__':
    if sys.argv[1] == 'study':
        study()
    else:
        make_zip(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6].split(','), sys.argv[7])
