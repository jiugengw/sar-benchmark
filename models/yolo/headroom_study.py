"""How much AP is left to win in each class, given how noisy the labels themselves are? Clean holdout, no GPU.

Every class loses AP in two ways: objects never found (100 - AP50) and boxes found but not tight enough for the strict
thresholds (AP50 - AP). The second has a floor that no model can beat: if the labels' own edges are noisy, even the
true box misses the strict thresholds. This study estimates that floor per class and how far we are above it.

1. Label noise, from twins: near-identical training images (sarbench/train_twins.json) whose boxes were drawn
   separately show how much two labels of the same object disagree. Per pair, the shared shift of all boxes (image
   registration) is removed, so only box-by-box disagreement counts. Each edge's difference, relative to the box size,
   has variance 2 sigma_label^2 (two noisy labels); robust estimate via the median absolute deviation.
2. Our error: each clean-holdout label against our best box (IoU >= 0.5), same edge measure, sigma_ours. It contains
   the label noise too: sigma_ours^2 = sigma_model^2 + sigma_label^2.
3. Tightness (AP / AP50) predicted from an edge noise sigma: perturb every edge of the class's holdout boxes with
   Gaussian noise of that size and average the share of boxes clearing each threshold 0.50-0.95, relative to 0.50.
   Check: the tightness predicted from sigma_ours should match our actual AP / AP50; if it does, the same simulation
   with sigma_label gives the best tightness any model could reach (the ceiling).
4. Loose-box headroom = AP50 x ceiling tightness - AP (at our current AP50). Missed-object loss = 100 - AP50.

    python models/yolo/headroom_study.py --raw results/pp-memorise-lr2/raw
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))

THRESHOLDS = np.linspace(0.5, 0.95, 10)
MIN_EDGES = 40  # fewer measured label edges than this: label noise reported as unknown


def xyxy(b):
    return np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], dtype=float)


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def relative_edges(label, other):
    """Edge differences other - label (x1, x2, y1, y2), each divided by the label's width or height."""
    w, h = label[2] - label[0], label[3] - label[1]
    d = other - label
    return [d[0] / w, d[2] / w, d[1] / h, d[3] / h]


def robust_sigma(values) -> float:
    values = np.asarray(values)
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def label_noise(data: Path) -> dict:
    """Per class: relative per-edge label noise sigma from separately drawn twin labels, and the number of edges."""
    coco = json.loads((data / 'train/instances.json').read_text())
    pairs = json.loads((ROOT / 'sarbench/train_twins.json').read_text())['pairs']
    by_file = {im['file_name']: im['id'] for im in coco['images']}
    names = {c['id']: c['name'] for c in coco['categories']}
    boxes = defaultdict(list)
    for a in coco['annotations']:
        boxes[a['image_id']].append((a['category_id'], xyxy(a['bbox'])))
    edges = defaultdict(list)
    for pair in pairs:
        a, b = (boxes[by_file[f]] for f in pair['images'])
        if not a or sorted(map(tuple, (x for _, x in a))) == sorted(map(tuple, (y for _, y in b))):
            continue  # identical boxes: one labelling copied, says nothing about noise
        matched = []
        for c, x in a:
            candidates = [y for d, y in b if d == c]
            if candidates:
                y = max(candidates, key=lambda y: iou(x, y))
                if iou(x, y) > 0.3:
                    matched.append((c, x, y))
        if len(matched) < 2:
            continue  # a lone box cannot separate image shift from label noise
        centres = np.array([((y[0] + y[2]) - (x[0] + x[2])) / 2 for _, x, y in matched]), \
            np.array([((y[1] + y[3]) - (x[1] + x[3])) / 2 for _, x, y in matched])
        shift = np.array([centres[0].mean(), centres[1].mean(), centres[0].mean(), centres[1].mean()])
        for c, x, y in matched:
            edges[names[c]].extend(relative_edges(x, y - shift))
    # the difference of two noisy labels has twice the variance of one label's noise
    return {c: (robust_sigma(e) / np.sqrt(2), len(e)) for c, e in edges.items()}


def tightness(sizes: np.ndarray, sigma: float, rng, draws: int = 200) -> float:
    """Mean over thresholds of P(IoU >= t) / P(IoU >= 0.5) when every edge of these boxes moves by N(0, sigma x size)."""
    w, h = sizes[:, 0], sizes[:, 1]
    truth = np.stack([np.zeros_like(w), np.zeros_like(h), w, h], 1)
    scale = np.stack([w, w, h, h], 1)
    noisy = truth[None] + rng.normal(0, sigma, (draws, *truth.shape)) * scale[None]
    noisy[..., 2] = np.maximum(noisy[..., 2], noisy[..., 0] + 1e-3)
    noisy[..., 3] = np.maximum(noisy[..., 3], noisy[..., 1] + 1e-3)
    iw = np.clip(np.minimum(noisy[..., 2], w) - np.maximum(noisy[..., 0], 0), 0, None)
    ih = np.clip(np.minimum(noisy[..., 3], h) - np.maximum(noisy[..., 1], 0), 0, None)
    inter = iw * ih
    ious = inter / (w * h + (noisy[..., 2] - noisy[..., 0]) * (noisy[..., 3] - noisy[..., 1]) - inter)
    share = np.array([(ious >= t).mean() for t in THRESHOLDS])
    return float(share.mean() / share[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', type=Path, default=ROOT / 'results/pp-memorise-lr2/raw',
                        help='holdout predictions before NMS (postprocess_study.py) and the holdout labels')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/course-data')
    args = parser.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from postprocess_study import apply
    from sarbench.twins import clean_image_ids

    gt_path = args.raw / 'instances_val_holdout.json'
    gt = json.loads(gt_path.read_text())
    clean = set(clean_image_ids(gt['images']))
    predictions = apply(json.loads((args.raw / 'predictions_val.json').read_text()), 'nms', 0.7)  # as submitted
    names = {c['id']: c['name'] for c in gt['categories']}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
        ev = COCOeval(coco_gt, coco_gt.loadRes([dict(p) for p in predictions]), 'bbox')
        ev.params.imgIds = sorted(clean)
        ev.evaluate()
        ev.accumulate()
    precision = ev.eval['precision'][:, :, :, 0, -1]  # [threshold, recall, class]
    ap_at = {names[c]: precision[:, :, k].mean(1) for k, c in enumerate(ev.params.catIds)}

    by_image = defaultdict(list)
    for p in predictions:
        by_image[(p['image_id'], p['category_id'])].append(xyxy(p['bbox']))
    ours, sizes = defaultdict(list), defaultdict(list)
    for a in gt['annotations']:
        if a['image_id'] not in clean:
            continue
        name, label = names[a['category_id']], xyxy(a['bbox'])
        sizes[name].append(a['bbox'][2:])
        candidates = by_image.get((a['image_id'], a['category_id']), [])
        if candidates:
            best = max(candidates, key=lambda p: iou(label, p))
            if iou(label, best) >= 0.5:
                ours[name].extend(relative_edges(label, best))

    noise = label_noise(args.data)
    rng = np.random.default_rng(0)
    rows = []
    for name in names.values():
        ap50, ap = 100 * ap_at[name][0], 100 * ap_at[name].mean()
        size = np.array(sizes[name], dtype=float)
        sigma_ours = robust_sigma(ours[name])
        sigma_label, n_label = noise.get(name, (float('nan'), 0))
        known = n_label >= MIN_EDGES
        ceiling = tightness(size, sigma_label, rng) if known else float('nan')
        row = {'class': name, 'labels': len(size), 'AP': ap, 'AP50': ap50, 'missed': 100 - ap50, 'loose': ap50 - ap,
               'tightness': ap / ap50, 'tightness_from_sigma_ours': tightness(size, sigma_ours, rng),
               'sigma_ours': sigma_ours, 'sigma_label': sigma_label, 'label_edges': n_label,
               'ceiling_tightness': ceiling, 'loose_headroom': ap50 * ceiling - ap if known else float('nan'),
               'median_size_px': float(np.median(np.sqrt(size.prod(1))))}
        rows.append(row)

    print(f"{'class':13s} {'n':>4s} {'size':>5s} {'AP':>5s} {'AP50':>5s} {'missed':>6s} {'loose':>5s} | "
          f"{'tight':>5s} {'check':>5s} | {'σ ours':>6s} {'σ label':>7s} {'(edges)':>7s} | {'ceil.':>5s} {'headroom':>8s}")
    for r in sorted(rows, key=lambda r: -r['loose']):
        print(f"{r['class']:13s} {r['labels']:4d} {r['median_size_px']:5.0f} {r['AP']:5.1f} {r['AP50']:5.1f} "
              f"{r['missed']:6.1f} {r['loose']:5.1f} | {r['tightness']:5.2f} {r['tightness_from_sigma_ours']:5.2f} | "
              f"{100 * r['sigma_ours']:5.1f}% {100 * r['sigma_label']:6.1f}% {r['label_edges']:7d} | "
              f"{r['ceiling_tightness']:5.2f} {r['loose_headroom']:8.1f}")
    print(f"mAP {np.mean([r['AP'] for r in rows]):.2f}; missed {np.mean([r['missed'] for r in rows]):.1f} and loose "
          f"{np.mean([r['loose'] for r in rows]):.1f} mAP points; loose headroom above the label-noise floor "
          f"{sum(r['loose_headroom'] for r in rows if r['loose_headroom'] == r['loose_headroom']) / len(rows):.1f} "
          f"mAP points from the classes where it is measurable")
    out = args.raw.parent / 'headroom_study.json'
    out.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
