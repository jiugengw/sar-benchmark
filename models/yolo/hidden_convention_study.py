"""Are our remaining box errors random, or do they follow a labelling convention we have not modelled?

Our boxes are unbiased per class and source, so the leftover edge error looks like noise. But SAR labels are often
drawn with rules a human cannot see in the image (from AIS positions, from optical images, including or excluding
sidelobes and shadow). If the error depends on something visible at test time, it is learnable and correctable.

For every holdout label matched by our box (same class, IoU >= 0.5), the target is each edge's offset from the label,
relative to the box size. The inputs are only things known at test time, from the predicted box and the image: class,
source width, position in the image, size, aspect, score, brightness inside / around the box, where the object's
bright centre sits inside the box, its orientation and elongation, and how many boxes are nearby. A gradient-boosted
model per edge is fitted on half of the holdout (twin groups together) and predicts the other half; three random
splits. Reported: how much of the edge error it explains (R^2), and, decisively, the clean and official-mix mAP after
moving every predicted edge by the predicted correction.

    python models/yolo/hidden_convention_study.py --results results/memorise-lr2-332326/one2many
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

FEATURES = ('class', 'source', 'cx', 'cy', 'log_w', 'log_h', 'log_aspect', 'score', 'mean_in', 'std_in', 'mean_ring',
            'contrast', 'peak_dx', 'peak_dy', 'centroid_dx', 'centroid_dy', 'angle_cos2', 'angle_sin2', 'elongation',
            'neighbours')


def box_features(image: np.ndarray, box, category, score, width, others) -> list[float]:
    """Test-time features of one predicted box (xywh, original pixels) in a grayscale image."""
    h_img, w_img = image.shape
    x, y, w, h = box
    x1, y1, x2, y2 = int(max(0, np.floor(x))), int(max(0, np.floor(y))), int(min(w_img, np.ceil(x + w))), \
        int(min(h_img, np.ceil(y + h)))
    patch = image[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)].astype(np.float32)
    pad = max(2, int(0.5 * max(w, h)))
    ring = image[max(0, y1 - pad):min(h_img, y2 + pad), max(0, x1 - pad):min(w_img, x2 + pad)].astype(np.float32)
    mean_in, std_in = float(patch.mean()), float(patch.std())
    ring_mean = float((ring.sum() - patch.sum()) / max(1, ring.size - patch.size))
    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    weights = np.clip(patch - np.median(patch), 0, None) + 1e-6
    cx_w, cy_w = float((weights * xx).sum() / weights.sum()), float((weights * yy).sum() / weights.sum())
    peak = np.unravel_index(int(np.argmax(patch)), patch.shape)
    # orientation and elongation from intensity-weighted second moments
    sxx = float((weights * (xx - cx_w) ** 2).sum() / weights.sum())
    syy = float((weights * (yy - cy_w) ** 2).sum() / weights.sum())
    sxy = float((weights * (xx - cx_w) * (yy - cy_w)).sum() / weights.sum())
    angle = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    spread = np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
    elongation = float(spread / ((sxx + syy) / 2 + 1e-6))
    width_box, height_box = max(patch.shape[1], 1), max(patch.shape[0], 1)
    return [category, 0 if width == 512 else 1 if width == 256 else 2, (x + w / 2) / w_img, (y + h / 2) / h_img,
            np.log(max(w, 1e-3)), np.log(max(h, 1e-3)), np.log(max(w, 1e-3) / max(h, 1e-3)), score, mean_in, std_in,
            ring_mean, mean_in - ring_mean, (peak[1] + 0.5) / width_box - 0.5, (peak[0] + 0.5) / height_box - 0.5,
            (cx_w + 0.5) / width_box - 0.5, (cy_w + 0.5) / height_box - 0.5, float(np.cos(2 * angle)),
            float(np.sin(2 * angle)), elongation, others]


def iou(a, b):
    w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=ROOT / 'results/memorise-lr2-332326/one2many')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--min-score', type=float, default=0.05, help='boxes below this score are left as they are')
    args = parser.parse_args()
    from PIL import Image
    from sklearn.ensemble import HistGradientBoostingRegressor
    from official_mix_score import NAMES, official_shares
    from ranking_study import twin_groups
    from resolution_probe import coco
    from sarbench.twins import clean_image_ids

    gt_path = args.results / 'instances_val_holdout.json'
    gt = json.loads(gt_path.read_text())
    images = {im['id']: im for im in gt['images']}
    clean = set(clean_image_ids(gt['images']))
    labels = defaultdict(list)
    for a in gt['annotations']:
        labels[(a['image_id'], a['category_id'])].append(a['bbox'])
    predictions = json.loads((args.results / 'predictions_val.json').read_text())
    by_image = defaultdict(list)
    for k, p in enumerate(predictions):
        by_image[p['image_id']].append(k)

    # features for every box worth correcting; targets where a label matches
    x = np.zeros((len(predictions), len(FEATURES)))
    target = np.full((len(predictions), 4), np.nan)
    active = np.zeros(len(predictions), bool)
    for image_id, members in by_image.items():
        info = images[image_id]
        pixels = np.asarray(Image.open(ROOT / 'data/course-data/train' / info['file_name']).convert('L'))
        boxes = [predictions[k]['bbox'] for k in members]
        for k, box in zip(members, boxes):
            p = predictions[k]
            if p['score'] < args.min_score:
                continue
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            near = sum(1 for b in boxes if b is not box and abs(b[0] + b[2] / 2 - cx) < 3 * box[2]
                       and abs(b[1] + b[3] / 2 - cy) < 3 * box[3])
            x[k] = box_features(pixels, box, p['category_id'], p['score'], info['width'], near)
            active[k] = True
            candidates = labels.get((image_id, p['category_id']), [])
            if candidates:
                best = max(candidates, key=lambda g: iou(box, g))
                if iou(box, best) >= 0.5:
                    target[k] = [(box[0] - best[0]) / box[2], (box[0] + box[2] - best[0] - best[2]) / box[2],
                                 (box[1] - best[1]) / box[3], (box[1] + box[3] - best[1] - best[3]) / box[3]]
    matched = active & ~np.isnan(target[:, 0])
    groups = twin_groups(gt['images'])
    group_of = np.array([groups[p['image_id']] for p in predictions])
    unique = np.array(sorted(set(group_of[active])))
    mix_zip = next(p for p in (ROOT / 'submissions/memorise-lr2-332326/memorise-lr2-332326.zip',
                               ROOT / 'nonsubmissions/191026/memorise-lr2-332326/memorise-lr2-332326.zip') if p.is_file())
    shares = official_shares(mix_zip, ROOT / 'data/course-data/val/images.json')

    def scores(boxes):
        widths = {i: images[i]['width'] for i in images}
        m, _ = coco(gt_path, boxes, clean)
        small = {i for i in clean if widths[i] == 256}
        _, ap_s = coco(gt_path, boxes, small)
        _, ap_l = coco(gt_path, boxes, clean - small)
        mixed = []
        for name in NAMES:
            s, l = ap_s.get(name, float('nan')), ap_l.get(name, float('nan'))
            mixed.append(l if s != s else s if l != l else shares[name] * s + (1 - shares[name]) * l)
        return 100 * m, 100 * float(np.mean(mixed))

    base_clean, base_mix = scores(predictions)
    print(f'{matched.sum()} matched boxes, {active.sum()} boxes with score >= {args.min_score}')
    print(f'edge error spread (robust sigma, % of box size): ' + ', '.join(
        f"{e} {100 * 1.4826 * np.median(np.abs(target[matched, j] - np.median(target[matched, j]))):.1f}"
        for j, e in enumerate(('left', 'right', 'top', 'bottom'))))
    print(f'baseline: clean mAP {base_clean:.2f}, official-mix {base_mix:.2f}', flush=True)
    r2s, results = [], []
    for seed in range(args.seeds):
        half = set(np.random.default_rng(seed).permutation(unique)[: len(unique) // 2])
        in_a = np.array([g in half for g in group_of])
        correction = np.zeros((len(predictions), 4))
        r2 = []
        for fit, apply in ((in_a, ~in_a), (~in_a, in_a)):
            for j in range(4):
                model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                      min_samples_leaf=30, l2_regularization=1.0,
                                                      categorical_features=[0, 1], random_state=0)
                train = fit & matched
                model.fit(x[train], np.clip(target[train, j], -0.5, 0.5))
                test = apply & active
                correction[test, j] = model.predict(x[test])
                held = apply & matched
                residual = target[held, j] - correction[held, j]
                r2.append(1 - np.var(residual) / np.var(target[held, j]))
        r2s.append(np.mean(r2))
        corrected = []
        for k, p in enumerate(predictions):
            if not active[k]:
                corrected.append(p)
                continue
            bx, by, bw, bh = p['bbox']
            l, r, t, b = correction[k]
            x1, x2 = bx - l * bw, bx + bw - r * bw
            y1, y2 = by - t * bh, by + bh - b * bh
            corrected.append({**p, 'bbox': [x1, y1, max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)]})
        clean_map, mix_map = scores(corrected)
        results.append((clean_map, mix_map))
        print(f'seed {seed}: edge-error variance explained {100 * np.mean(r2):.1f}%; corrected clean mAP {clean_map:.2f} '
              f'({clean_map - base_clean:+.2f}), official-mix {mix_map:.2f} ({mix_map - base_mix:+.2f})', flush=True)
    clean_mean, mix_mean = np.mean([r[0] for r in results]), np.mean([r[1] for r in results])
    print(f'mean: variance explained {100 * np.mean(r2s):.1f}%; clean {clean_mean - base_clean:+.2f}, '
          f'official-mix {mix_mean - base_mix:+.2f} (seed-to-seed noise of whole runs is about 0.1)')
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
                                          l2_regularization=1.0, categorical_features=[0, 1], random_state=0)
    from sklearn.inspection import permutation_importance
    importance = defaultdict(float)
    for j in range(4):
        model.fit(x[matched], np.clip(target[matched, j], -0.5, 0.5))
        result = permutation_importance(model, x[matched], np.clip(target[matched, j], -0.5, 0.5), n_repeats=3,
                                        random_state=0)
        for name, value in zip(FEATURES, result.importances_mean):
            importance[name] += value / 4
    print('most informative inputs (in-sample, permutation importance): ' + ', '.join(
        f'{k} {v:.3f}' for k, v in sorted(importance.items(), key=lambda kv: -kv[1])[:6]))


if __name__ == '__main__':
    main()
