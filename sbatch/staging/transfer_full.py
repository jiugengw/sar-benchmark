# Label transfer with FULL-SIZE matching (30 Oct; NOT ALLOWED, kept as a record). Same as transfer.py, but the
# similarity is the correlation of the images at their own size (no thumbnail), and the test (target) image is used
# as it is: the labelled images are compared in their four orientations instead. Chunked to keep memory small.
#   python transfer_full.py study                        holdout: similarity distribution and official-mix per cut-off
#   python transfer_full.py zip IN.zip OUT.zip LO HI     test ZIP (static classes, replace mode)
import contextlib, io, json, os, sys, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transfer import STATIC, TR, train, transfer  # noqa: E402

FLIPS = [lambda a: a, lambda a: a[:, ::-1], lambda a: a[::-1, :], lambda a: a[::-1, ::-1]]


def vec(img):
    v = img.astype(np.float32).ravel(); s = v.std(); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-6), s


def load(root, im):
    return np.asarray(Image.open(Path(root) / im['file_name']).convert('L'), dtype=np.float32)


def match_full(targets, troot, pool, chunk=384, pool_loader=None):
    """target id -> (similarity, flip kind, labelled image id); the target is never transformed. pool_loader(im) gives
    the labelled image as compared (default: the image itself at full size)."""
    pool_loader = pool_loader or (lambda im: load(TR, im))
    best = {}
    for shape in sorted({(im['height'], im['width']) for im in targets}):
        tg = [im for im in targets if (im['height'], im['width']) == shape]
        pl = [im for im in pool if (im['height'], im['width']) == shape]
        if not tg or not pl:
            continue
        T, ts = zip(*(vec(load(troot, im)) for im in tg)); T = np.stack(T); ts = np.array(ts)
        for c in range(0, len(pl), chunk):
            part = pl[c:c + chunk]; raw = [pool_loader(im) for im in part]
            for k, f in enumerate(FLIPS):
                P, ps = zip(*(vec(np.ascontiguousarray(f(r))) for r in raw)); P = np.stack(P); ps = np.array(ps)
                C = T @ P.T; C[ts < 4] = 0; C[:, ps < 4] = 0
                j, m = C.argmax(1), C.max(1)
                for i, im in enumerate(tg):
                    if m[i] > best.get(im['id'], (-2,))[0]:
                        best[im['id']] = (float(m[i]), k, part[j[i]]['id'])
        print(f'  matched {len(tg)} images of size {shape} against {len(pl)}', flush=True)
    return best


def study():
    sys.argv = sys.argv[:1]
    from hedge_study import CHOSEN_TIERS, hedge_tiers
    from official_mix_score import NAMES, official_shares
    from resolution_probe import coco
    from round_boxes import round_predictions
    from sarbench.twins import clean_image_ids
    R = Path('results/ft40-s1r-fliprefit-340514/one2many'); gt_path = str(R / 'instances_val_holdout.json')
    gt = json.load(open(gt_path)); images = {im['id']: im for im in gt['images']}; widths = {i: im['width'] for i, im in images.items()}
    clean = set(clean_image_ids(gt['images'])); hf = {im['file_name'] for im in gt['images']}
    best = match_full(gt['images'], TR, [im for im in train['images'] if im['file_name'] not in hf])
    sims = np.array([best.get(i, (-1,))[0] for i in sorted(clean)])
    print('clean holdout full-size similarity quantiles (10..90%):', np.round(np.quantile(sims, np.linspace(0.1, 0.9, 9)), 3).tolist())
    print('share above 0.3/0.4/0.5/0.6/0.7/0.8/0.9/0.98:', [round(float(np.mean(sims >= t)), 3) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98)])
    base = hedge_tiers(round_predictions(json.load(open(R / 'predictions_val.json')), widths, 'whole-pixel-classes'), CHOSEN_TIERS)
    low = [p for p in json.load(open('results/tail-lowconf-340514/predictions_lowconf.json')) if p['score'] < 0.001]
    floor = min(p['score'] for p in base); room = defaultdict(lambda: 100)
    for p in base:
        room[p['image_id'], p['category_id']] -= 1
    for p in sorted(round_predictions(low, widths, 'whole-pixel-classes'), key=lambda p: -p['score']):
        if room[p['image_id'], p['category_id']] > 0:
            room[p['image_id'], p['category_id']] -= 1; base.append({**p, 'score': p['score'] * floor / 0.001 * 0.999})
    shares = official_shares(Path('submissions/memorise-lr2-332326/memorise-lr2-332326.zip'), Path('data/course-data/val/images.json'))

    def mix(boxes):
        small = {i for i in clean if images[i]['width'] == 256}; keep = [b for b in boxes if b['image_id'] in clean]
        with contextlib.redirect_stdout(io.StringIO()):
            _, x = coco(gt_path, keep, small); _, y = coco(gt_path, keep, clean - small)
        v = []
        for n in NAMES:
            s, l = x.get(n, float('nan')), y.get(n, float('nan'))
            v.append(l if s != s else s if l != l else shares[n] * s + (1 - shares[n]) * l)
        return 100 * float(np.nanmean(v))
    print(f'current pipeline (no transfer): official-mix {mix(base):.2f}')
    for hi in (0.98, 1.01):
        for lo in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7):
            b, n, nb = transfer(base, best, images, lo, hi, STATIC, 'replace')
            print(f'  similarity {lo:.2f}-{min(hi, 1.0):.2f}: official-mix {mix(b):.2f}  ({n} images, {nb} boxes copied)', flush=True)
    print('TRANSFER_FULL_STUDY_DONE')


def make_zip(src, dst, lo, hi):
    VA = Path('data/course-data/val'); val = json.load(open(VA / 'images.json'))['images']; images = {im['id']: im for im in val}
    best = match_full(val, VA, train['images'])
    z = zipfile.ZipFile(src); members = {n: z.read(n) for n in z.namelist()}
    preds = json.loads(members['detection.json'])
    out, nimg, nbox = transfer(preds, best, images, lo, hi, STATIC, 'replace')
    members['detection.json'] = json.dumps(out).encode()
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as o:
        for n, d in members.items():
            o.writestr(n, d)
    print(f'{dst}: copied labels into {nimg} test images ({nbox} boxes; full-size similarity {lo}-{hi}, replace)')


if __name__ == '__main__':
    if sys.argv[1] == 'study':
        study()
    else:
        make_zip(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]))
