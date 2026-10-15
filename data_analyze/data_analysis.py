#!/usr/bin/env python3
"""Pre-preprocessing data analysis for the CS701 SAR course data.

Looks at images and boxes only to decide on preprocessing / augmentation / anchors.
It never uses val or test labels (there are none); val images are only used for
distribution statistics (is val like train?), which is just looking, not training.

Usage (from the repo root, or Colab after extracting the tarballs):
    python data_analyze/data_analysis.py --data data/course-data

Outputs (default data_analyze/results/): report.txt, summary.json, fig*.png
Runtime: ~2-5 min for the full scan (reads every image once) + pixel analysis on a sample.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402

CLASSES = ['aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil_tank', 'playground', 'ship', 'wind_turbine']
CODE_MEAN, CODE_STD = 0.2526, 0.2133  # constants hard-coded in sarbench/data.py
REPORT, SUMMARY = [], {}
POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    REPORT.append(s)


def H(title):
    P('\n' + '=' * 78 + f'\n{title}\n' + '=' * 78)


# ----------------------------------------------------------------------------- loading
def resolve(split_root, file_name):
    for p in (split_root / file_name, split_root / 'images' / file_name,
              split_root / 'images' / Path(file_name).name):
        if p.exists():
            return p
    raise FileNotFoundError(f'{file_name} not under {split_root}')


def scan(job):
    """One full read of an image: header facts, exact 256-bin histogram, hashes."""
    split, sr, name = job
    p = resolve(sr, name)
    with Image.open(p) as im:
        mode, (w, h) = im.mode, im.size
        rgb_equal = None
        if mode in ('RGB', 'RGBA'):
            a3 = np.asarray(im.convert('RGB'))
            rgb_equal = bool((a3[..., 0] == a3[..., 1]).all() and (a3[..., 1] == a3[..., 2]).all())
        arr = np.asarray(im.convert('L'))
    hist = np.bincount(arr.ravel(), minlength=256)
    sm = np.asarray(Image.fromarray(arr).resize((16, 16), Image.BOX), dtype=np.float32)
    return dict(split=split, name=name, path=str(p), mode=mode, w=w, h=h, ext=p.suffix.lower(),
                hist=hist, md5=hashlib.md5(arr.tobytes()).hexdigest(),
                ah=np.packbits((sm > sm.mean()).ravel()), thumb=sm.ravel(), rgb_equal=rgb_equal)


def hist_stats(h):
    n = h.sum()
    v = np.arange(256) / 255.0
    mean = float((h * v).sum() / n)
    std = float(np.sqrt(max((h * v * v).sum() / n - mean ** 2, 0)))
    cdf = np.cumsum(h) / n
    q = lambda p: float(np.searchsorted(cdf, p / 100.0) / 255.0)  # noqa: E731
    return dict(mean=mean, std=std, p1=q(1), p50=q(50), p99=q(99), p999=q(99.9),
                sat=float(h[255] / n), zero=float(h[0] / n))


# ----------------------------------------------------------------------------- pixel-level (sample)
def analyse_pixels(path, win=7, big=15):
    """Local mean/std windows (noise model) and speckle lag correlations for one image."""
    with Image.open(path) as im:
        f = np.asarray(im.convert('L'), dtype=np.float32) / 255.0
    m = uniform_filter(f, win)
    s = np.sqrt(np.maximum(uniform_filter(f * f, win) - m * m, 0))
    ms, ss = m[::win, ::win].ravel(), s[::win, ::win].ravel()
    # speckle correlation: residual relative to a larger local mean, low-texture pixels only
    mb = uniform_filter(f, big)
    e = f / np.maximum(mb, 1e-3) - 1.0
    cv = s / np.maximum(m, 1e-3)
    ok = (m > 0.03) & (cv < np.percentile(cv[m > 0.03], 30)) if (m > 0.03).sum() > 1000 else np.zeros_like(f, bool)
    out = dict(m=ms, s=ss, corr={})
    for name, (sl_a, sl_b) in {'x1': ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                              'y1': ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                              'x2': ((slice(None), slice(2, None)), (slice(None), slice(None, -2))),
                              'y2': ((slice(2, None), slice(None)), (slice(None, -2), slice(None)))}.items():
        mk = ok[sl_a] & ok[sl_b]
        if mk.sum() > 2000:
            out['corr'][name] = float(np.corrcoef(e[sl_a][mk], e[sl_b][mk])[0, 1])
    return out


def box_contrast(path, boxes):
    """Per box (native pixels, xyxy): target-to-clutter ratio and mean contrast vs a surrounding ring."""
    with Image.open(path) as im:
        a = np.asarray(im.convert('L'), dtype=np.float32)
    H_, W_ = a.shape
    res = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
        x2, y2 = int(min(W_, max(x1 + 1, np.ceil(x2)))), int(min(H_, max(y1 + 1, np.ceil(y2))))
        pad = max(4, int(0.5 * max(x2 - x1, y2 - y1)))
        X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(W_, x2 + pad), min(H_, y2 + pad)
        outer = a[Y1:Y2, X1:X2]
        mask = np.ones(outer.shape, bool)
        mask[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = False
        if mask.sum() < 8:
            continue
        inside = a[y1:y2, x1:x2]
        ring = outer[mask].mean()
        res.append((20 * np.log10((inside.max() + 1) / (ring + 1)), (inside.mean() + 1) / (ring + 1)))
    return res


# ----------------------------------------------------------------------------- boxes
def anchors(sizes, ratios):
    out = []
    for scales in sizes:
        for s in scales:
            for r in ratios:  # torchvision: h/w = ratio, area = s^2
                hr = r ** 0.5
                out.append((s / hr, s * hr))
    return np.array(out)


def best_iou(w, h, A):
    inter = np.minimum(w[:, None], A[None, :, 0]) * np.minimum(h[:, None], A[None, :, 1])
    union = (w * h)[:, None] + (A[:, 0] * A[:, 1])[None] - inter
    return (inter / union).max(1)


def xyxy_iou(b):
    x1, y1, x2, y2 = b.T
    iw = np.maximum(0, np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]))
    ih = np.maximum(0, np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]))
    inter = iw * ih
    area = (x2 - x1) * (y2 - y1)
    return inter / np.maximum(area[:, None] + area[None] - inter, 1e-9)


def cramers_v(table):
    chi2 = stats.chi2_contingency(table)[0]
    n = table.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))) if min(table.shape) > 1 else 0.0


def eta_squared(groups):
    allv = np.concatenate(groups)
    ss_tot = ((allv - allv.mean()) ** 2).sum()
    ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    return float(ss_b / ss_tot) if ss_tot > 0 else 0.0


def bp(ax, data, labels, **kw):
    try:
        return ax.boxplot(data, tick_labels=labels, **kw)  # matplotlib >= 3.9
    except TypeError:
        return ax.boxplot(data, labels=labels, **kw)


def size_group(w, h):
    return f'{w}x{h}' if (w, h) in ((256, 256), (512, 512)) else 'other'


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=Path('data/course-data'))
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'results')
    ap.add_argument('--sample', type=int, default=500, help='train images for pixel-level analysis')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- load metadata
    tr_root, va_root = args.data / 'train', args.data / 'val'
    with open(tr_root / 'labels.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    coco = json.loads((tr_root / 'instances.json').read_text())
    coco_imgs = {im['file_name']: im for im in coco['images']}
    by_id = {im['id']: im for im in coco['images']}
    label_of = {r['file_name']: int(r['label']) for r in rows}
    val_imgs = json.loads((va_root / 'images.json').read_text())['images'] if (va_root / 'images.json').exists() else []

    H('0. WHAT WAS LOADED')
    P(f'train: {len(rows)} labelled images, {len(coco["annotations"])} boxes | val: {len(val_imgs)} images (no labels)')

    # ---- full scan
    jobs = [('train', tr_root, r['file_name']) for r in rows] + [('val', va_root, im['file_name']) for im in val_imgs]
    with ThreadPoolExecutor(args.workers) as ex:
        recs = list(ex.map(scan, jobs))
    tr = [r for r in recs if r['split'] == 'train']
    va = [r for r in recs if r['split'] == 'val']
    for r in tr:
        r['label'] = label_of[r['name']]
        r['stats'] = hist_stats(r['hist'])
    for r in va:
        r['stats'] = hist_stats(r['hist'])

    # =============================================================== 1. integrity
    H('1. FILE FORMAT AND INTEGRITY')
    for nm, rs in (('train', tr), ('val', va)):
        if not rs:
            continue
        P(f'[{nm}] modes {dict(Counter(r["mode"] for r in rs))} | extensions {dict(Counter(r["ext"] for r in rs))} '
          f'| sizes {dict(Counter(size_group(r["w"], r["h"]) for r in rs))}')
        rgb = [r for r in rs if r['rgb_equal'] is not None]
        if rgb:
            P(f'[{nm}] {sum(r["rgb_equal"] for r in rgb)}/{len(rgb)} multi-channel files have identical channels (true grey)')
    bad_size = [r['name'] for r in tr if (r['w'], r['h']) != (coco_imgs[r['name']]['width'], coco_imgs[r['name']]['height'])]
    mism = Counter()
    n_noboxes = 0
    ann_by_img = defaultdict(list)
    for a in coco['annotations']:
        ann_by_img[a['image_id']].append(a)
    for r in tr:
        anns = ann_by_img[coco_imgs[r['name']]['id']]
        n_noboxes += len(anns) == 0
        mism['category != label+1'] += sum(a['category_id'] != r['label'] + 1 for a in anns)
    wh = np.array([a['bbox'][2:] for a in coco['annotations']], float)
    xy = np.array([a['bbox'][:2] for a in coco['annotations']], float)
    imw = np.array([by_id[a['image_id']]['width'] for a in coco['annotations']], float)
    imh = np.array([by_id[a['image_id']]['height'] for a in coco['annotations']], float)
    P(f'size in json != size on disk: {len(bad_size)} | images with zero boxes: {n_noboxes} | '
      f'boxes with w<=0 or h<=0: {int(((wh <= 0).any(1)).sum())} | '
      f'boxes extending outside image: {int(((xy[:, 0] + wh[:, 0] > imw + 1) | (xy[:, 1] + wh[:, 1] > imh + 1) | (xy < -1).any(1)).sum())} | '
      f'box category != image label+1: {mism["category != label+1"]}')
    SUMMARY['integrity'] = dict(size_mismatch=len(bad_size), zero_box_images=n_noboxes, **mism)

    # duplicates inside train (they would leak between your own train/holdout split and inflate local scores)
    md5c = Counter(r['md5'] for r in tr)
    exact = {k for k, v in md5c.items() if v > 1}
    ah = np.stack([r['ah'] for r in tr]).astype(np.uint8)
    pairs = []
    for i0 in range(0, len(tr), 128):
        x = ah[i0:i0 + 128, None, :] ^ ah[None, :, :]
        d = POPCOUNT[x].sum(-1)
        ii, jj = np.nonzero(d <= 8)  # <= 8 of 256 bits differ
        for a, b in zip(ii, jj):
            i = i0 + a
            if i < b and np.corrcoef(tr[i]['thumb'], tr[b]['thumb'])[0, 1] > 0.95:  # verify: hash AND thumbnail agree
                pairs.append((i, b))
    conflict = sum(tr[i]['label'] != tr[j]['label'] for i, j in pairs)
    P(f'exact duplicate groups in train: {len(exact)} | near-duplicate pairs (16x16 hash <=8 bits and thumbnail corr >0.95): {len(pairs)} '
      f'(involving {len({i for p in pairs for i in p})} images, label conflicts: {conflict})')
    for i, j in pairs[:5]:
        P(f'   e.g. {tr[i]["name"]}  ~  {tr[j]["name"]}')
    SUMMARY['duplicates'] = dict(exact_groups=len(exact), near_pairs=len(pairs), label_conflicts=int(conflict))

    # =============================================================== 2. intensity
    H('2. INTENSITY DISTRIBUTION  (decides clipping / gamma / log / normalisation)')
    gh = {nm: sum(r['hist'] for r in rs) for nm, rs in (('train', tr), ('val', va)) if rs}
    for nm, h in gh.items():
        s = hist_stats(h)
        P(f'[{nm}] pixels: mean {s["mean"]:.4f} std {s["std"]:.4f} | p1 {s["p1"]*255:.0f} p50 {s["p50"]*255:.0f} '
          f'p99 {s["p99"]*255:.0f} p99.9 {s["p999"]*255:.0f} (0-255) | exactly 0: {s["zero"]*100:.2f}% exactly 255: {s["sat"]*100:.3f}%')
    P(f'normalisation constants in data.py: mean {CODE_MEAN} std {CODE_STD} (those were computed after resizing to 512; native-res values above should be close)')
    s_tr = np.array([[r['stats'][k] for k in ('mean', 'std', 'p99', 'sat', 'zero')] for r in tr])
    P(f'per-image mean brightness: p5 {np.percentile(s_tr[:,0],5):.3f}  median {np.median(s_tr[:,0]):.3f}  p95 {np.percentile(s_tr[:,0],95):.3f}'
      f'  | images with >1% saturated pixels: {(s_tr[:,3]>0.01).mean()*100:.1f}% | images with >50% black pixels: {(s_tr[:,4]>0.5).mean()*100:.1f}%')
    SUMMARY['intensity'] = {nm: hist_stats(h) for nm, h in gh.items()}

    # train vs val shift
    shift = {}
    if va:
        s_va = np.array([[r['stats'][k] for k in ('mean', 'std', 'p99', 'sat', 'zero')] for r in va])
        for i, k in enumerate(('mean', 'std', 'p99', 'sat', 'zero')):
            ks = stats.ks_2samp(s_tr[:, i], s_va[:, i])
            shift[k] = float(ks.statistic)
        szt = Counter(size_group(r['w'], r['h']) for r in tr)
        szv = Counter(size_group(r['w'], r['h']) for r in va)
        P('train vs val KS statistic per image stat (0 = same, >0.1 = noticeable): ' +
          ', '.join(f'{k} {v:.3f}' for k, v in shift.items()))
        P('image-size mix  train: ' + ', '.join(f'{k} {v/len(tr)*100:.1f}%' for k, v in szt.items()) +
          '  | val: ' + ', '.join(f'{k} {v/len(va)*100:.1f}%' for k, v in szv.items()))
    SUMMARY['train_val_ks'] = shift

    # class-level confounders (could a model cheat from brightness or size alone?)
    cls_groups = [np.array([r['stats']['mean'] for r in tr if r['label'] == c]) for c in range(9)]
    eta_mean = eta_squared(cls_groups)
    sizes_tr = sorted({size_group(r['w'], r['h']) for r in tr})
    ct = np.array([[sum(1 for r in tr if r['label'] == c and size_group(r['w'], r['h']) == sg) for sg in sizes_tr]
                   for c in range(9)])
    cv_size = cramers_v(ct)
    P(f'class vs per-image mean brightness: eta^2 = {eta_mean:.3f} (share of brightness variance explained by class)')
    P(f'class vs image size: Cramer V = {cv_size:.3f}   (0 = independent, >0.3 = size leaks the class)')
    P('class x image-size counts:  ' + '  '.join(sizes_tr))
    for c in range(9):
        P(f'   {CLASSES[c]:13s} ' + '  '.join(f'{v:6d}' for v in ct[c]))
    SUMMARY['confounders'] = dict(eta2_brightness=eta_mean, cramers_v_size=cv_size)

    # =============================================================== 3. noise model (sample)
    H('3. SPECKLE / NOISE MODEL  (decides despeckling domain, whitening, log transform)')
    idx = rng.choice(len(tr), size=min(args.sample, len(tr)), replace=False)
    sample = [tr[i] for i in idx]
    with ThreadPoolExecutor(args.workers) as ex:
        pix = list(ex.map(lambda r: analyse_pixels(r['path']), sample))
    m_all = np.concatenate([p['m'] for p in pix])
    s_all = np.concatenate([p['s'] for p in pix])
    keep = m_all > 0.02
    m_k, s_k = m_all[keep], s_all[keep]
    edges = np.quantile(m_k, np.linspace(0, 1, 17))
    mc, floor = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (m_k >= a) & (m_k <= b)
        if sel.sum() > 50:
            mc.append(np.median(m_k[sel]))
            floor.append(np.percentile(s_k[sel], 20))  # low-texture floor ~ speckle
    mc, floor = np.array(mc), np.array(floor)
    okf = floor > 0
    expo, c0 = np.polyfit(np.log(mc[okf]), np.log(floor[okf]), 1)
    cvw = s_k / m_k
    mid = (m_k > 0.05) & (m_k < 0.6)
    P(f'local-window (7x7) std vs mean, speckle floor (20th pct of std per mean-bin): std ~ mean^{expo:.2f}')
    P('   exponent ~1  -> multiplicative noise in a linear (amplitude/intensity) scale: std grows with mean')
    P('   exponent ~0  -> roughly additive noise (log/dB-like or heavily compressed 8-bit mapping)')
    P('   in between   -> partially compressed scale (sqrt / gamma stretch)')
    P(f'coefficient of variation in the calmest windows (mean>0.05): p5 {np.percentile(cvw[m_k>0.05],5):.2f}  '
      f'p10 {np.percentile(cvw[m_k>0.05],10):.2f}  p25 {np.percentile(cvw[m_k>0.05],25):.2f}')
    P('   rough reference for fully developed speckle in a LINEAR scale: single-look amplitude ~0.52, single-look intensity ~1.0,'
      ' L-look intensity ~1/sqrt(L). 8-bit files are usually stretched, so use this only as a clue.')
    corr_by = defaultdict(lambda: defaultdict(list))
    for r, p in zip(sample, pix):
        for k, v in p['corr'].items():
            corr_by[size_group(r['w'], r['h'])][k].append(v)
    P('speckle lag correlation (low-texture pixels, mean over images)   [x = along columns, y = along rows]')
    corr_mean = {}
    for g, d in corr_by.items():
        corr_mean[g] = {k: float(np.mean(v)) for k, v in d.items()}
        P(f'   {g:8s} (n={len(d.get("x1", []))})  lag1 x {corr_mean[g].get("x1", float("nan")):.2f}  y {corr_mean[g].get("y1", float("nan")):.2f}'
          f'  | lag2 x {corr_mean[g].get("x2", float("nan")):.2f}  y {corr_mean[g].get("y2", float("nan")):.2f}')
    P('   ~0.0-0.1 = nearly independent pixels | 0.3+ = correlated speckle or an upsampled image (filters that assume i.i.d. speckle suffer)')
    SUMMARY['noise'] = dict(std_vs_mean_exponent=float(expo), cv_p10=float(np.percentile(cvw[m_k > 0.05], 10)), lag_corr=corr_mean)

    # =============================================================== 4. boxes
    H('4. BOXES  (decides small-object handling, anchors, NMS)')
    ann = coco['annotations']
    A = np.array([[a['image_id'], a['category_id'], *a['bbox'], by_id[a['image_id']]['width'],
                   by_id[a['image_id']]['height']] for a in ann], float)
    bw, bh, W_, H_ = A[:, 4], A[:, 5], A[:, 6], A[:, 7]
    w512, h512 = bw * 512 / W_, bh * 512 / H_  # what the model sees with geometry=stretch
    sq512, sqnat = np.sqrt(w512 * h512), np.sqrt(bw * bh)
    P(f'boxes: {len(A)} | per-image count: min {min(len(v) for v in ann_by_img.values())} '
      f'median {np.median([len(v) for v in ann_by_img.values()]):.0f} max {max(len(v) for v in ann_by_img.values())} '
      f'| images with exactly 1 box: {np.mean([len(v) == 1 for v in ann_by_img.values()])*100:.0f}%')
    P(f'box size at 512 input (sqrt(w*h)): p10 {np.percentile(sq512,10):.1f}  median {np.median(sq512):.1f}  p90 {np.percentile(sq512,90):.1f}  max {sq512.max():.0f}'
      f'   [slide says median 15.9, 50% below one 16-px ViT patch]')
    P(f'   max side at 512: median {np.median(np.maximum(w512,h512)):.1f} | native resolution sqrt(w*h): median {np.median(sqnat):.1f}')
    for t in (8, 12, 16, 24, 32):
        P(f'   boxes smaller than {t:2d}px at 512: {(sq512 < t).mean()*100:5.1f}%')
    asp = np.log2(np.maximum(w512, 1e-3) / np.maximum(h512, 1e-3))
    P(f'aspect ratio w/h: median {2**np.median(asp):.2f} | p5 {2**np.percentile(asp,5):.2f} p95 {2**np.percentile(asp,95):.2f} | '
      f'more elongated than 3:1 either way: {(np.abs(asp) > np.log2(3)).mean()*100:.1f}%')
    border = ((A[:, 2] <= 1) | (A[:, 3] <= 1) | (A[:, 2] + bw >= W_ - 1) | (A[:, 3] + bh >= H_ - 1)).mean()
    P(f'boxes touching an image border: {border*100:.1f}%')
    P('per class (512 scale):  boxes   imgs  median-size  p10   p90   <16px')
    for c in range(9):
        sel = A[:, 1] == c + 1
        nimg = sum(1 for r in tr if r['label'] == c)
        P(f'   {CLASSES[c]:13s} {sel.sum():6d} {nimg:6d} {np.median(sq512[sel]):9.1f} {np.percentile(sq512[sel],10):6.1f} '
          f'{np.percentile(sq512[sel],90):6.1f} {(sq512[sel]<16).mean()*100:6.1f}%')
    # crowding
    max_ov, cover = [], []
    for iid, anns in ann_by_img.items():
        im = by_id[iid]
        b = np.array([[a['bbox'][0] * 512 / im['width'], a['bbox'][1] * 512 / im['height'],
                       (a['bbox'][0] + a['bbox'][2]) * 512 / im['width'], (a['bbox'][1] + a['bbox'][3]) * 512 / im['height']]
                      for a in anns])
        cover.append(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).sum() / 512 ** 2)
        if len(b) > 1:
            iou = xyxy_iou(b)
            np.fill_diagonal(iou, 0)
            max_ov += list(iou.max(1))
    max_ov = np.array(max_ov)
    P(f'crowding: boxes overlapping another box with IoU>0.3: {(max_ov>0.3).mean()*100:.1f}%, IoU>0.5: {(max_ov>0.5).mean()*100:.1f}% (of boxes in multi-box images)'
      f' | box area / image area: median {np.median(cover)*100:.1f}%  p95 {np.percentile(cover,95)*100:.1f}%')
    P('   (NMS in train.py uses 0.5 at RoI level and 0.7 at RPN; IoU>0.5 overlaps between true boxes get suppressed)')
    # anchor coverage
    sizes_def = ((8, 11), (16, 23), (32, 45), (64, 91), (128, 181))
    cov = {}
    for nm, ratios in (('default (0.5,1,2)', (0.5, 1.0, 2.0)), ('wide (0.2..5)', (0.2, 0.5, 1.0, 2.0, 5.0))):
        bi = best_iou(w512, h512, anchors(sizes_def, ratios))
        cov[nm] = bi
        P(f'anchor coverage [{nm}] best centred-shape IoU per box: median {np.median(bi):.2f} | >=0.7: {(bi>=0.7).mean()*100:.1f}% '
          f'| 0.5-0.7: {((bi>=0.5)&(bi<0.7)).mean()*100:.1f}% | <0.5: {(bi<0.5).mean()*100:.1f}%')
    P('   upper bound (anchors on a stride grid are offset from the box centre, so real IoUs are lower, esp. for tiny boxes)')
    SUMMARY['boxes'] = dict(n=len(A), median_sqrt_area_512=float(np.median(sq512)), frac_lt16=float((sq512 < 16).mean()),
                            frac_lt8=float((sq512 < 8).mean()), frac_overlap_gt03=float((max_ov > 0.3).mean()),
                            anchor_cov_lt05_default=float((cov['default (0.5,1,2)'] < 0.5).mean()))

    # =============================================================== 5. contrast
    H('5. OBJECT CONTRAST  (baseline to check a filter does not erase targets)')
    by_img_boxes = {}
    for r in sample:
        im = coco_imgs[r['name']]
        by_img_boxes[r['name']] = [(a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3], a['category_id'])
                                   for a in ann_by_img[im['id']]]
    tcr, ratio, bcls, bsz = [], [], [], []
    for r in sample:
        bl = by_img_boxes[r['name']][:30]
        for (res, b) in zip(box_contrast(r['path'], [x[:4] for x in bl]), bl):
            tcr.append(res[0]); ratio.append(res[1]); bcls.append(b[4] - 1)
            bsz.append(np.sqrt((b[2] - b[0]) * (b[3] - b[1])) * 512 / r['w'])
    tcr, ratio, bcls, bsz = map(np.array, (tcr, ratio, bcls, bsz))
    P(f'{len(tcr)} boxes measured. TCR = 20log10(max in box / mean of surrounding ring), in dB')
    P(f'   TCR median {np.median(tcr):.1f} dB | p10 {np.percentile(tcr,10):.1f} | p90 {np.percentile(tcr,90):.1f}')
    P(f'   mean-in-box / mean-around: median {np.median(ratio):.2f} | boxes with ratio < 1.2 (hard, barely brighter than surroundings): {(ratio<1.2).mean()*100:.1f}%')
    P('   by class (median TCR dB | median ratio):')
    for c in range(9):
        sel = bcls == c
        if sel.sum():
            P(f'      {CLASSES[c]:13s} {np.median(tcr[sel]):6.1f} | {np.median(ratio[sel]):5.2f}   (n={sel.sum()})')
    for lo, hi in ((0, 8), (8, 16), (16, 32), (32, 1e9)):
        sel = (bsz >= lo) & (bsz < hi)
        if sel.sum():
            P(f'   size {lo:>2}-{hi if hi<1e9 else "inf"}px: median TCR {np.median(tcr[sel]):5.1f} dB  ratio {np.median(ratio[sel]):4.2f} (n={sel.sum()})')
    SUMMARY['contrast'] = dict(median_tcr_db=float(np.median(tcr)), frac_ratio_lt12=float((ratio < 1.2).mean()))

    # =============================================================== figures
    plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3})
    col = {'train': '#1f3a93', 'val': '#e08a00'}

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    cnt_img = np.bincount([r['label'] for r in tr], minlength=9)
    cnt_box = np.bincount((A[:, 1] - 1).astype(int), minlength=9)
    ax[0, 0].bar(range(9), cnt_img, color=col['train']); ax[0, 0].set_title('images per class')
    ax[0, 1].bar(range(9), cnt_box, color=col['train']); ax[0, 1].set_title('boxes per class')
    for a_ in ax[0, :2]:
        a_.set_xticks(range(9)); a_.set_xticklabels(CLASSES, rotation=60, ha='right')
    ax[0, 2].hist([len(v) for v in ann_by_img.values()], bins=np.arange(0.5, 42.5), color=col['train']); ax[0, 2].set_yscale('log')
    ax[0, 2].set_title('boxes per image')
    w_ = 0.8 / len(sizes_tr)
    for k, sg in enumerate(sizes_tr):
        ax[1, 0].bar(np.arange(9) + k * w_, ct[:, k], w_, label=sg)
    ax[1, 0].set_xticks(np.arange(9) + 0.4 - w_ / 2); ax[1, 0].set_xticklabels(CLASSES, rotation=60, ha='right')
    ax[1, 0].legend(); ax[1, 0].set_title(f'class x image size (Cramer V {cv_size:.2f})')
    ax[1, 1].hist(np.array(cover) * 100, bins=40, color=col['train']); ax[1, 1].set_title('box area as % of image'); ax[1, 1].set_yscale('log')
    ax[1, 2].hist(max_ov, bins=40, color=col['train']); ax[1, 2].set_yscale('log'); ax[1, 2].set_title('max IoU with another box (same image)')
    fig.tight_layout(); fig.savefig(args.out / 'fig1_dataset_and_classes.png'); plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for nm, h in gh.items():
        ax[0, 0].plot(np.arange(256), h / h.sum(), color=col[nm], label=nm)
    ax[0, 0].set_yscale('log'); ax[0, 0].set_title('pixel histogram (log y). A spike at 255 = saturation'); ax[0, 0].legend()
    for i, (k, t) in enumerate((('mean', 'per-image mean'), ('std', 'per-image std'), ('p99', 'per-image 99th pct'))):
        a_ = (ax[0, 1], ax[0, 2], ax[1, 0])[i]
        a_.hist(s_tr[:, i], bins=50, alpha=0.6, density=True, color=col['train'], label='train')
        if va:
            a_.hist(s_va[:, i], bins=50, alpha=0.6, density=True, color=col['val'], label='val')
        a_.set_title(f'{t} (KS {shift.get(k, 0):.2f})'); a_.legend()
    bp(ax[1, 1], cls_groups, CLASSES); ax[1, 1].tick_params(axis='x', rotation=60)
    ax[1, 1].set_title(f'per-image mean by class (eta^2 {eta_mean:.2f})')
    ax[1, 2].hist(s_tr[:, 3] * 100 + 1e-4, bins=np.logspace(-4, 2, 50), color=col['train']); ax[1, 2].set_xscale('log')
    ax[1, 2].set_title('% saturated pixels per image (log x)')
    fig.tight_layout(); fig.savefig(args.out / 'fig2_intensity.png'); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    sub = rng.choice(len(m_k), size=min(300000, len(m_k)), replace=False)
    ax[0, 0].hexbin(m_k[sub], s_k[sub], gridsize=60, bins='log', extent=(0, 1, 0, 0.5), cmap='viridis')
    xs = np.linspace(mc.min(), mc.max(), 50)
    ax[0, 0].plot(xs, np.exp(c0) * xs ** expo, 'r-', lw=2, label=f'speckle floor: std ~ mean^{expo:.2f}')
    ax[0, 0].plot(mc, floor, 'wo', ms=4); ax[0, 0].set_xlabel('local mean (7x7)'); ax[0, 0].set_ylabel('local std'); ax[0, 0].legend()
    ax[0, 0].set_title('noise model: std vs mean (exponent 1 = multiplicative)')
    ax[0, 1].hist(np.clip(cvw[m_k > 0.05], 0, 3), bins=80, color=col['train']); ax[0, 1].set_title('coefficient of variation per window (calm windows = left edge)')
    groups = sorted(corr_mean)
    for k, g in enumerate(groups):
        ax[1, 0].bar(np.arange(2) + k * 0.8 / len(groups), [corr_mean[g].get('x1', 0), corr_mean[g].get('y1', 0)], 0.8 / len(groups), label=g)
    ax[1, 0].set_xticks([0.2, 1.2]); ax[1, 0].set_xticklabels(['lag-1 along x', 'lag-1 along y']); ax[1, 0].legend()
    ax[1, 0].set_title('speckle lag-1 correlation by image size')
    ax[1, 1].loglog(mc[okf], floor[okf], 'o-'); ax[1, 1].set_title('speckle floor, log-log (slope = exponent)'); ax[1, 1].set_xlabel('local mean')
    fig.tight_layout(); fig.savefig(args.out / 'fig3_noise_model.png'); plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[0, 0].hist(sq512, bins=np.logspace(0, 2.6, 60), color=col['train']); ax[0, 0].set_xscale('log')
    for t in (8, 16, 32):
        ax[0, 0].axvline(t, color='r', ls='--')
    ax[0, 0].set_title('box size sqrt(w*h) at 512 (lines: 8/16/32 px)')
    ax[0, 1].scatter(w512[::3], h512[::3], s=2, alpha=0.3)
    ad = anchors(sizes_def, (0.5, 1.0, 2.0))
    ax[0, 1].scatter(ad[:, 0], ad[:, 1], c='r', marker='x', s=25, label='default anchors')
    ax[0, 1].set_xscale('log'); ax[0, 1].set_yscale('log'); ax[0, 1].legend(); ax[0, 1].set_title('box w vs h at 512 and anchor shapes')
    ax[0, 2].hist(asp, bins=60, color=col['train']); ax[0, 2].set_title('log2(w/h): 0 square, +-1 = 2:1')
    for nm, bi in cov.items():
        ax[1, 0].hist(bi, bins=40, alpha=0.5, label=nm)
    ax[1, 0].axvline(0.5, color='k', ls='--'); ax[1, 0].axvline(0.7, color='k', ls=':'); ax[1, 0].legend(); ax[1, 0].set_title('best-anchor IoU per box')
    bp(ax[1, 1], [sq512[A[:, 1] == c + 1] for c in range(9)], CLASSES, showfliers=False); ax[1, 1].set_yscale('log')
    ax[1, 1].tick_params(axis='x', rotation=60); ax[1, 1].axhline(16, color='r', ls='--'); ax[1, 1].set_title('box size by class (512 scale)')
    ax[1, 2].axis('off')
    fig.tight_layout(); fig.savefig(args.out / 'fig4_boxes.png'); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    bp(ax[0], [tcr[bcls == c] for c in range(9) if (bcls == c).any()], [CLASSES[c] for c in range(9) if (bcls == c).any()], showfliers=False)
    ax[0].tick_params(axis='x', rotation=60); ax[0].set_title('target-to-clutter ratio (dB) by class')
    bins = [(0, 8), (8, 16), (16, 32), (32, 1e9)]
    bp(ax[1], [tcr[(bsz >= lo) & (bsz < hi)] for lo, hi in bins if ((bsz >= lo) & (bsz < hi)).any()],
       [f'{lo}-{int(hi) if hi<1e9 else "inf"}' for lo, hi in bins if ((bsz >= lo) & (bsz < hi)).any()], showfliers=False)
    ax[1].set_title('TCR (dB) by box size at 512')
    ax[2].hist(np.clip(ratio, 0, 5), bins=60, color=col['train']); ax[2].axvline(1.2, color='r', ls='--'); ax[2].set_title('mean-in-box / mean-around (red: 1.2)')
    fig.tight_layout(); fig.savefig(args.out / 'fig5_contrast.png'); plt.close(fig)

    # gallery of tiny boxes + contact sheet
    tiny = [(r, b) for r in sample for b in by_img_boxes[r['name']] if np.sqrt((b[2] - b[0]) * (b[3] - b[1])) * 512 / r['w'] < 16]
    rng.shuffle(tiny)
    fig, ax = plt.subplots(4, 8, figsize=(16, 8.4))
    for a_, (r, b) in zip(ax.ravel(), tiny[:32]):
        with Image.open(r['path']) as im:
            g = np.asarray(im.convert('L'))
        cx, cy, pad = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 16
        x0, y0 = int(max(0, cx - pad)), int(max(0, cy - pad))
        a_.imshow(g[y0:int(cy + pad), x0:int(cx + pad)], cmap='gray', vmin=0, vmax=255, interpolation='nearest',
                  extent=(x0, x0 + g[y0:int(cy + pad), x0:int(cx + pad)].shape[1], y0 + g[y0:int(cy + pad), x0:int(cx + pad)].shape[0], y0))
        a_.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec='yellow', lw=0.8))
        a_.set_title(CLASSES[b[4] - 1], fontsize=8); a_.axis('off')
    for a_ in ax.ravel()[len(tiny[:32]):]:
        a_.axis('off')
    fig.suptitle('random boxes smaller than 16 px at 512 (native pixels, 32x32 window around each)'); fig.tight_layout()
    fig.savefig(args.out / 'fig6_tiny_boxes.png'); plt.close(fig)

    fig, ax = plt.subplots(3, 4, figsize=(16, 12))
    for a_, i in zip(ax.ravel(), rng.choice(len(tr), 12, replace=False)):
        r = tr[i]
        with Image.open(r['path']) as im:
            a_.imshow(np.asarray(im.convert('L')), cmap='gray', vmin=0, vmax=255)
        for b in by_img_boxes.get(r['name']) or [(a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3], 0)
                                                  for a in ann_by_img[coco_imgs[r['name']]['id']]]:
            a_.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec='yellow', lw=0.8))
        a_.set_title(f'{CLASSES[r["label"]]}  {r["w"]}x{r["h"]}', fontsize=9); a_.axis('off')
    fig.tight_layout(); fig.savefig(args.out / 'fig7_contact_sheet.png'); plt.close(fig)

    # =============================================================== decision hints
    H('6. WHAT THIS SUGGESTS (hints, not verdicts: confirm with an ablation)')
    s = hist_stats(gh['train'])
    if s['sat'] > 0.005:
        P(f'- {s["sat"]*100:.2f}% of pixels are saturated at 255: the bright tail is already clipped; percentile-clipping the top adds little, gamma/log mainly helps the dark/mid range.')
    else:
        P('- Very few saturated pixels: the bright tail is intact. A log/gamma or high-percentile clip is meaningful and safe to test.')
    if s['p99'] < 0.6 and s['p50'] < 0.2:
        P(f'- The image is mostly dark (median {s["p50"]*255:.0f}, p99 {s["p99"]*255:.0f}): a gamma < 1 or log transform would spread the useful range. Test it.')
    if expo > 0.7:
        P(f'- Noise std scales with local mean (exponent {expo:.2f}): multiplicative speckle in a nearly linear scale. A log transform (makes speckle additive) is a principled option to test.')
    elif expo < 0.35:
        P(f'- Noise std barely depends on the mean (exponent {expo:.2f}): the 8-bit mapping already looks log/dB-like. A further log transform is unlikely to help; skip it.')
    else:
        P(f'- Noise partially depends on the mean (exponent {expo:.2f}): a compressed (sqrt/gamma-like) mapping. A mild gamma is the most you should test.')
    big_corr = max([v.get('x1', 0) for v in corr_mean.values()] + [v.get('y1', 0) for v in corr_mean.values()])
    if big_corr > 0.3:
        P('- Speckle is spatially correlated (lag-1 > 0.3). Despeckling filters that assume i.i.d. speckle (Lee, Kuan) are less effective; '
          'compare a 256x256 group vs a 512x512 group: a much higher value in one suggests that group was upsampled.')
    if len(corr_mean) > 1:
        vals = [v.get('x1', 0) for v in corr_mean.values()]
        if max(vals) - min(vals) > 0.15:
            P('- Lag correlation differs a lot between image-size groups: the groups have different resolution/processing history; consider whether stretching 256->512 hides that from the model.')
    if (sq512 < 16).mean() > 0.4:
        P(f'- {(sq512<16).mean()*100:.0f}% of boxes are under 16 px at 512: keep any filter window <=3x3 (or none); a 7x7 filter is as large as the object.')
    lowc = (ratio < 1.2).mean()
    if lowc > 0.15:
        P(f'- {lowc*100:.0f}% of boxes are barely brighter than their surroundings: smoothing will hurt these; compare TCR/ratio before and after any filter and reject filters that lower them.')
    if (cov['default (0.5,1,2)'] < 0.5).mean() > 0.05:
        P(f'- {(cov["default (0.5,1,2)"]<0.5).mean()*100:.1f}% of boxes have no anchor with IoU>=0.5 (default ratios): try --anchor-profile wide.')
    if cv_size > 0.3:
        P(f'- Class and image size are strongly linked (Cramer V {cv_size:.2f}): the model can read class from resampling blur even after stretching to 512. Treat classification gains with suspicion; consider a size-aware holdout.')
    if eta_mean > 0.25:
        P(f'- Brightness alone explains {eta_mean*100:.0f}% of between-class variance: global intensity transforms change a class cue, not just noise.')
    if shift and max(shift.values()) > 0.1:
        P(f'- Train and val differ noticeably in {[k for k,v in shift.items() if v>0.1]} (KS>0.1): your local holdout may be optimistic; weigh Codabench val more.')
    if SUMMARY['duplicates']['near_pairs'] or SUMMARY['duplicates']['exact_groups']:
        P('- Duplicates / near-duplicates exist inside train: a random 90/10 split puts copies on both sides and inflates local scores; group duplicates before splitting.')
    P(f'- Normalisation: train mean/std at native res = {hist_stats(gh["train"])["mean"]:.4f}/{hist_stats(gh["train"])["std"]:.4f} vs {CODE_MEAN}/{CODE_STD} in data.py. '
      'Recompute after ANY intensity transform and apply it identically in make_submission.py.')

    (args.out / 'report.txt').write_text('\n'.join(REPORT))
    (args.out / 'summary.json').write_text(json.dumps(SUMMARY, indent=2, default=float))
    P(f'\nSaved report.txt, summary.json and fig1-7 to {args.out}')


if __name__ == '__main__':
    main()
