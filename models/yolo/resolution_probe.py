"""Diagnostic only (never a submission): is input resolution what limits the tiny objects?

Our best model scores oil tanks 63.6 AP in 256 px images (which reach the network upscaled x2) but 21.6 in 512 px
images, and wind turbines are 8 px across. If resolution is the limit, the same trained model should gain on those
objects when the holdout is simply fed larger. The rules cap the input at 512, so this only measures the lever; using
it would need an architecture that works at a finer internal stride on the same 512 input.

Same weights, one-to-many head, NMS 0.7 and confidence 0.001 as the submission code; Ultralytics letterboxes each
original holdout image to the given size and maps boxes back to original pixels. Scores per class on the clean holdout,
split by image source (256 vs 512 px wide).

    python models/yolo/resolution_probe.py --weights runs/ft40-memorise-lr2-332326/train/weights/best.pt
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))

HOLDOUT = ROOT / 'results/yolo26m-100-p2-328339/seed_0/instances_val_holdout.json'


def coco(gt_path, predictions, image_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        ev = COCOeval(gt, gt.loadRes([dict(p) for p in predictions]), 'bbox')
        ev.params.imgIds = sorted(image_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    precision = ev.eval['precision'][:, :, :, 0, -1]
    names = {c['id']: c['name'] for c in gt.dataset['categories']}
    per_class = {}
    for k, c in enumerate(ev.params.catIds):
        values = precision[:, :, k]
        per_class[names[c]] = float(values[values > -1].mean()) if (values > -1).any() else float('nan')
    return float(ev.stats[0]), per_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--sizes', default='512,768,1024')
    parser.add_argument('--tag', default='resolution-probe')
    args = parser.parse_args()
    from detector_support import load_detector
    from sarbench.twins import clean_image_ids

    gt = json.loads(HOLDOUT.read_text())
    clean = set(clean_image_ids(gt['images']))
    width = {im['id']: im['width'] for im in gt['images']}
    by_path = {str(ROOT / 'data/course-data/train' / im['file_name']): im['id'] for im in gt['images']}
    detector = load_detector(args.weights, 'yolo')
    detector.model.end2end = False  # one-to-many head + NMS, as submitted
    out = ROOT / 'results' / args.tag
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for size in map(int, args.sizes.split(',')):
        predictions = []
        paths = sorted(by_path)
        for start in range(0, len(paths), 64):
            for result in detector.predict(source=paths[start:start + 64], imgsz=size, batch=16, device=0,
                                           conf=0.001, iou=0.7, nms=True, max_det=300, verbose=False, save=False):
                image_id = by_path[str(Path(result.path))]
                for (x1, y1, x2, y2), c, s in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                                  result.boxes.conf.tolist()):
                    predictions.append({'image_id': image_id, 'category_id': int(c) + 1,
                                        'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(s)})
        (out / f'predictions_{size}.json').write_text(json.dumps(predictions))
        row = {}
        for subset, ids in (('clean', clean), ('clean 512px', {i for i in clean if width[i] == 512}),
                            ('clean 256px', {i for i in clean if width[i] == 256})):
            row[subset] = coco(HOLDOUT, predictions, ids)
        report[size] = row
        print(f'== input {size}px', flush=True)
        for subset, (m, per_class) in row.items():
            print(f'{subset:12s} mAP {100 * m:5.2f} | ' + ' '.join(
                f'{k[:6]} {100 * v:5.1f}' for k, v in per_class.items()), flush=True)
    (out / 'resolution_probe.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
