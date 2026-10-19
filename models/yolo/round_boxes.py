"""Snap predicted box edges to whole pixels where the labels are drawn on whole pixels.

Labels of some classes and sources sit exactly on pixel edges (aircraft, car and ship everywhere; bridge, harbor and oil
tank in the 256 px images; bridge gains nothing), so a predicted edge 0.3 px off loses IoU it cannot win back; rounding it to the nearest
pixel recovers that whenever the prediction is within half a pixel. Deterministic post-processing of the one model's
boxes, applied after NMS; scores and classifications are untouched.

    python models/yolo/round_boxes.py study results/memorise-lr2-332326/one2many          # holdout gain per mode
    python models/yolo/round_boxes.py zip in.zip out.zip --mode whole-pixel-classes       # rewrite a submission
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# COCO category ids: 1 aircraft, 4 car, 5 harbor, 6 oil_tank, 8 ship
ALWAYS = {1, 4, 8}
ON_256 = {5, 6}  # bridge left out: rounding cost it 0.2 AP on the holdout
MODES = ('none', 'whole-pixel-classes', 'all')


def snap(box: list[float]) -> list[float]:
    x1, y1 = round(box[0]), round(box[1])
    x2, y2 = round(box[0] + box[2]), round(box[1] + box[3])
    return [float(x1), float(y1), float(max(x2 - x1, 1)), float(max(y2 - y1, 1))]


def round_predictions(predictions: list[dict], widths: dict[int, int], mode: str) -> list[dict]:
    out = []
    for p in predictions:
        c, w = p['category_id'], widths[p['image_id']]
        if mode == 'all' or (mode == 'whole-pixel-classes' and (c in ALWAYS or (c in ON_256 and w == 256))):
            p = {**p, 'bbox': snap(p['bbox'])}
        out.append(p)
    return out


def study(results: Path) -> None:
    from sarbench.metrics import detection_metrics
    from sarbench.twins import clean_image_ids

    gt_path = results / 'instances_val_holdout.json'
    images = json.loads(gt_path.read_text())['images']
    widths = {im['id']: im['width'] for im in images}
    full, clean = [im['id'] for im in images], clean_image_ids(images)
    predictions = json.loads((results / 'predictions_val.json').read_text())
    print(f"{'mode':22s} {'full mAP':>8s} {'clean mAP':>9s}  clean per class")
    for mode in MODES:
        boxes = round_predictions(predictions, widths, mode)
        f, c = detection_metrics(str(gt_path), boxes, full), detection_metrics(str(gt_path), boxes, clean)
        per = ' '.join(f"{k[:5]} {100 * v:5.1f}" for k, v in c['per_class_AP'].items())
        print(f"{mode:22s} {100 * f['mAP']:8.2f} {100 * c['mAP']:9.2f}  {per}", flush=True)


def rewrite_zip(source: Path, target: Path, mode: str, images_json: Path) -> None:
    widths = {im['id']: im['width'] for im in json.loads(images_json.read_text())['images']}
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    predictions = json.loads(members['detection.json'])
    members['detection.json'] = json.dumps(round_predictions(predictions, widths, mode)).encode()
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    print(f'{target}: {len(predictions)} boxes, mode {mode}; other files byte-for-byte unchanged')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('study')
    s.add_argument('results', type=Path, help='folder with predictions_val.json and instances_val_holdout.json')
    z = sub.add_parser('zip')
    z.add_argument('source', type=Path)
    z.add_argument('target', type=Path)
    z.add_argument('--mode', choices=MODES[1:], default='whole-pixel-classes')
    z.add_argument('--images', type=Path, default=ROOT / 'data/course-data/val/images.json',
                   help='image list with widths for the ZIP\'s images (official validation by default)')
    args = parser.parse_args()
    if args.command == 'study':
        study(args.results)
    else:
        rewrite_zip(args.source, args.target, args.mode, args.images)


if __name__ == '__main__':
    main()
