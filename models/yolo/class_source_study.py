"""Why does one class score so low on one image source? (Written for bridges on 512 px images: 5 AP, vs 22 at 256 px.)

On the clean holdout, for the class's images of the chosen width, it reports:
- AP at each IoU threshold from 0.50 to 0.95: is the loss at loose overlap (wrong or missing objects) or only at strict
  overlap (box edges)?
- every label's best overlap with a same-class prediction, in bands: found well, found loosely, found badly, missed;
  and whether a missed label is covered by a box of another class instead;
- the shape of the labels (size, aspect, boxes per image) next to the same class on the other image width and in the
  training split, and the predicted / labelled size ratios of found objects;
- false alarms: confident same-class boxes far from every label, in the class's own images and in other images;
- the scores of hits and false alarms (AP also drops when false alarms outrank hits).
With --figure, it draws a grid of the class's images: labels green, predictions (score >= 0.3) red with their score.

    python models/yolo/class_source_study.py --results results/ft40-m4repeat-beta8-333674/one2many --figure bridges.png
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
DATA = ROOT / 'data/course-data/train'


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, 4) x (m, 4) xywh boxes -> (n, m) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    w = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = w * h
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter + 1e-9)


def ap_per_threshold(gt_path: Path, predictions: list[dict], image_ids, category: int) -> np.ndarray:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        ev = COCOeval(gt, gt.loadRes([dict(p) for p in predictions]), 'bbox')
        ev.params.imgIds, ev.params.catIds = sorted(image_ids), [category]
        ev.evaluate()
        ev.accumulate()
    precision = ev.eval['precision'][:, :, 0, 0, -1]  # IoU thresholds x recall points
    return np.array([p[p > -1].mean() if (p > -1).any() else np.nan for p in precision])


def describe(boxes: np.ndarray) -> str:
    if len(boxes) == 0:
        return 'none'
    w, h = boxes[:, 2], boxes[:, 3]
    long, short = np.maximum(w, h), np.minimum(w, h)
    q = lambda v: f'{np.percentile(v, 25):.0f}/{np.median(v):.0f}/{np.percentile(v, 75):.0f}'
    return (f'{len(boxes)} boxes; long side {q(long)} px, short side {q(short)} px (25%/median/75%), '
            f'long/short {np.median(long / np.maximum(short, 1e-3)):.1f}, wider than tall {100 * np.mean(w > h):.0f}%')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=ROOT / 'results/ft40-m4repeat-beta8-333674/one2many')
    parser.add_argument('--category', default='bridge')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--figure', type=Path, help='write a grid of example images here (PNG)')
    parser.add_argument('--figure-images', type=int, default=16)
    args = parser.parse_args()
    from sarbench.twins import clean_image_ids

    gt_path = args.results / 'instances_val_holdout.json'
    gt = json.loads(gt_path.read_text())
    category = next(c['id'] for c in gt['categories'] if c['name'] == args.category)
    names = {c['id']: c['name'] for c in gt['categories']}
    images = {im['id']: im for im in gt['images']}
    clean = set(clean_image_ids(gt['images']))
    labels = defaultdict(list)
    for a in gt['annotations']:
        labels[a['image_id']].append(a)
    image_class = {i: max({a['category_id'] for a in v}, key=lambda c: sum(a['category_id'] == c for a in v))
                   for i, v in labels.items()}
    predictions = json.loads((args.results / 'predictions_val.json').read_text())
    by_image = defaultdict(list)
    for p in predictions:
        by_image[p['image_id']].append(p)

    def subset(width):
        return sorted(i for i in clean if images[i]['width'] == width and image_class.get(i) == category)

    own = subset(args.width)
    other_width = 256 if args.width == 512 else 512
    print(f'{args.category}: {len(own)} clean holdout images at {args.width} px, {len(subset(other_width))} at '
          f'{other_width} px')

    # 1. AP by IoU threshold
    thresholds = np.linspace(0.5, 0.95, 10)
    print('\nAP by IoU threshold   ' + ' '.join(f'{t:5.2f}' for t in thresholds) + '   mean')
    width_ids = {w: {i for i in clean if images[i]['width'] == w} for w in (args.width, other_width)}
    for w in (args.width, other_width):
        ap = ap_per_threshold(gt_path, predictions, width_ids[w], category)
        print(f'  {w} px images       ' + ' '.join(f'{100 * v:5.1f}' for v in ap) + f'  {100 * np.nanmean(ap):5.1f}')

    # 2. every label's best same-class overlap, and what covers the missed ones
    bands = [(0.75, 'found well (IoU >= 0.75)'), (0.5, 'found loosely (0.5-0.75)'), (0.3, 'found badly (0.3-0.5)'),
             (0.1, 'barely touched (0.1-0.3)'), (-1, 'missed (< 0.1)')]
    best, best_score, ratios, offsets, other_cover, gt_boxes = [], [], [], [], defaultdict(int), []
    for i in own:
        g = np.array([a['bbox'] for a in labels[i] if a['category_id'] == category], dtype=float)
        gt_boxes.extend(g)
        same = [p for p in by_image[i] if p['category_id'] == category and p['score'] >= 0.05]
        p_boxes = np.array([p['bbox'] for p in same], dtype=float).reshape(-1, 4)
        overlap = iou_xywh(g, p_boxes)
        for k, row in enumerate(overlap):
            j = int(row.argmax()) if len(row) else -1
            value = float(row[j]) if j >= 0 else 0.0
            best.append(value)
            best_score.append(same[j]['score'] if j >= 0 else 0.0)
            if value >= 0.3:
                pb, gb = p_boxes[j], g[k]
                ratios.append((pb[2] / gb[2], pb[3] / gb[3]))
                offsets.append(((pb[0] + pb[2] / 2 - gb[0] - gb[2] / 2) / gb[2],
                                (pb[1] + pb[3] / 2 - gb[1] - gb[3] / 2) / gb[3]))
            if value < 0.5:
                others = [p for p in by_image[i] if p['category_id'] != category and p['score'] >= 0.05]
                o = iou_xywh(g[k:k + 1], np.array([p['bbox'] for p in others], dtype=float).reshape(-1, 4))
                if o.size and o.max() >= 0.5:
                    other_cover[names[others[int(o.argmax())]['category_id']]] += 1
    best, best_score = np.array(best), np.array(best_score)
    print(f'\n{len(best)} {args.category} labels in those images: best same-class prediction (score >= 0.05)')
    low = 1.01
    for edge, name in bands:
        mask = (best >= edge) & (best < low)
        print(f'  {name:26s} {mask.sum():4d}  ({100 * mask.mean():4.1f}%)  median score of that box '
              f'{np.median(best_score[mask]) if mask.any() else float("nan"):.2f}')
        low = edge
    if other_cover:
        print('  labels under 0.5 covered (IoU >= 0.5) by another class instead: ' +
              ', '.join(f'{k} {v}' for k, v in sorted(other_cover.items(), key=lambda kv: -kv[1])))

    # 3. label shape, here and elsewhere
    print('\nlabel shapes')
    print(f'  holdout {args.width} px:   {describe(np.array(gt_boxes))}')
    others_box = np.array([a['bbox'] for i in subset(other_width) for a in labels[i] if a['category_id'] == category])
    print(f'  holdout {other_width} px:   {describe(others_box)}')
    train = json.loads((DATA / 'instances.json').read_text())
    held = {im['file_name'] for im in gt['images']}
    train_width = {im['id']: im['width'] for im in train['images'] if im['file_name'] not in held}
    for w in (args.width, other_width):
        b = np.array([a['bbox'] for a in train['annotations'] if a['category_id'] == category
                      and train_width.get(a['image_id']) == w])
        n_img = len({a['image_id'] for a in train['annotations'] if a['category_id'] == category
                     and train_width.get(a['image_id']) == w})
        print(f'  training {w} px:  {n_img} images, {len(b) / max(n_img, 1):.1f} per image; {describe(b)}')
    per_image = [sum(a['category_id'] == category for a in labels[i]) for i in own]
    print(f'  holdout {args.width} px: {np.mean(per_image):.1f} labels per image (max {max(per_image)})')
    if ratios:
        r, o = np.array(ratios), np.array(offsets)
        print(f'\nfound objects (IoU >= 0.3), predicted / labelled: width {np.median(r[:, 0]):.2f} '
              f'({np.percentile(r[:, 0], 25):.2f}-{np.percentile(r[:, 0], 75):.2f}), height {np.median(r[:, 1]):.2f} '
              f'({np.percentile(r[:, 1], 25):.2f}-{np.percentile(r[:, 1], 75):.2f}); centre shift '
              f'{100 * np.median(np.abs(o[:, 0])):.0f}% of width, {100 * np.median(np.abs(o[:, 1])):.0f}% of height')

    # 4. false alarms and the scores of hits
    hits, alarms_own, alarms_other = [], [], 0
    for i in clean:
        if images[i]['width'] != args.width:
            continue
        g = np.array([a['bbox'] for a in labels[i] if a['category_id'] == category], dtype=float).reshape(-1, 4)
        for p in by_image[i]:
            if p['category_id'] != category or p['score'] < 0.05:
                continue
            o = iou_xywh(np.array([p['bbox']], dtype=float), g)
            value = o.max() if o.size else 0.0
            if value >= 0.5:
                hits.append(p['score'])
            elif value < 0.1:
                if i in own:
                    alarms_own.append(p['score'])
                elif p['score'] >= 0.3:
                    alarms_other += 1
    hits, alarms_own = np.array(hits), np.array(alarms_own)
    print(f'\n{args.category} boxes (score >= 0.05) at {args.width} px: {len(hits)} hits (IoU >= 0.5), '
          f'{len(alarms_own)} false alarms (IoU < 0.1) in {args.category} images, '
          f'{alarms_other} confident (>= 0.3) ones in other classes\' images')
    for t in (0.3, 0.5, 0.7):
        print(f'  score >= {t}: {np.sum(hits >= t)} hits, {np.sum(alarms_own >= t)} false alarms in own images')

    if args.figure:
        draw(args, own, images, labels, by_image, category)


def draw(args, own, images, labels, by_image, category) -> None:
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(0)
    chosen = sorted(rng.choice(own, size=min(args.figure_images, len(own)), replace=False).tolist())
    tile, cols = 384, 4
    rows = (len(chosen) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tile, rows * tile), 'white')
    for n, i in enumerate(chosen):
        im = Image.open(DATA / images[i]['file_name']).convert('RGB')
        scale = tile / max(im.size)
        im = im.resize((round(im.width * scale), round(im.height * scale)))
        pen = ImageDraw.Draw(im)
        for a in labels[i]:
            x, y, w, h = (v * scale for v in a['bbox'])
            pen.rectangle([x, y, x + w, y + h], outline=(0, 255, 0) if a['category_id'] == category else (0, 160, 255),
                          width=2)
        for p in by_image[i]:
            if p['score'] >= 0.3:
                x, y, w, h = (v * scale for v in p['bbox'])
                pen.rectangle([x, y, x + w, y + h], outline=(255, 40, 40), width=1)
                pen.text((x + 2, y + 1), f"{p['score']:.2f}" + ('' if p['category_id'] == category else '*'),
                         fill=(255, 40, 40))
        pen.text((4, 4), images[i]['file_name'][-28:], fill=(255, 255, 0))
        sheet.paste(im, ((n % cols) * tile, (n // cols) * tile))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.figure)
    print(f'\nfigure: {args.figure} (green labels, red predictions >= 0.3, * = another class)')


if __name__ == '__main__':
    main()
