"""Write class counts, image shapes, and idealized anchor-shape coverage (no GPU needed)."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def shape_iou(w, h, aw, ah):
    intersection = min(w, aw) * min(h, ah)
    return intersection / (w * h + aw * ah - intersection)


def coverage(boxes, sizes, ratios):
    anchors = [(size / math.sqrt(ratio), size * math.sqrt(ratio))
               for size in sizes for ratio in ratios]
    best = [max(shape_iou(w, h, aw, ah) for aw, ah in anchors) for w, h in boxes]
    return {'boxes': len(best), 'mean_best_centered_iou': sum(best) / len(best) if best else None,
            'fraction_at_least_0.5': sum(x >= .5 for x in best) / len(best) if best else None,
            'fraction_at_least_0.7': sum(x >= .7 for x in best) / len(best) if best else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.data / 'train'
    coco = json.loads((root / 'instances.json').read_text())
    with (root / 'labels.csv').open() as file:
        rows = list(csv.DictReader(file))
    images = {image['id']: image for image in coco['images']}
    categories = {category['id']: category['name'] for category in coco['categories']}
    shapes = defaultdict(list)
    for ann in coco['annotations']:
        image = images[ann['image_id']]
        w, h = ann['bbox'][2:]
        if w > 0 and h > 0:
            shapes[categories[ann['category_id']]].append((w * 512 / image['width'], h * 512 / image['height']))
    profiles = {
        'default': ((8, 11, 16, 23, 32, 45, 64, 91, 128, 181), (.5, 1, 2)),
        'wide': ((8, 11, 16, 23, 32, 45, 64, 91, 128, 181), (.2, .5, 1, 2, 5)),
        'large': ((8, 11, 16, 23, 32, 45, 64, 91, 181, 256), (.5, 1, 2)),
    }
    aspects = [max(im['width'] / im['height'], im['height'] / im['width']) for im in images.values()]
    report = {
        'note': 'Full labeled-training data summary; no validation labels used for training. '
                'Centered shape IoU assumes perfect anchor centers and ignores stride/rounding; '
                'this diagnoses shapes, not actual RPN recall. Padding changes geometry separately.',
        'classification_image_counts': dict(Counter(row['label'] for row in rows)),
        'category_ids': categories,
        'box_counts': {name: len(boxes) for name, boxes in shapes.items()},
        'images': len(images),
        'fraction_nonsquare': sum(x > 1.01 for x in aspects) / len(aspects),
        'fraction_aspect_over_1.5': sum(x > 1.5 for x in aspects) / len(aspects),
        'anchor_shape_coverage': {profile: {name: coverage(boxes, sizes, ratios)
                                          for name, boxes in shapes.items()}
                                  for profile, (sizes, ratios) in profiles.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f'Data diagnostics saved to {args.out}', flush=True)


if __name__ == '__main__':
    main()
