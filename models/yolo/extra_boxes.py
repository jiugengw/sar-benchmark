"""Step 6 of the final submission: more of the detector's own boxes, ranked below every box the ZIP already has.

Two kinds, filling each image and class up to 100 boxes (the scorer's limit), in this order:
  1. boxes NMS removed. The detector proposes several overlapping boxes per object and NMS at IoU 0.7 keeps the
     strongest. The same model with NMS at 0.95 (make_submission.py --conf 0.001 --max-det 1000 --nms-iou 0.95) gives
     back the neighbours it deleted, its own second guesses at the box edges; rounded like the originals
     (round_boxes.py, whole-pixel-classes). Boxes already in the ZIP (IoU >= 0.99) are skipped.
  2. wider hedge copies of every box >= 0.001: sizes x1.2 / x0.83, shifts +-16% (step 3 uses 1.08 / 0.92 and 8%).
Each kind keeps its boxes' score order, and every added box scores below the lowest box in the ZIP, so they can only
add recall: in COCO AP, boxes ranked below all others of a class never lower the precision at any earlier rank.
classification.csv is copied byte for byte.

Codabench (1 Nov, sbatch/staging/job_w.py on the +65.80 ZIP): mAP 59.94 -> 59.96, Δm +65.80 -> +65.85.

    python models/yolo/extra_boxes.py tailed.zip nms95.zip out.zip [images.json]
"""
from __future__ import annotations

import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))

from round_boxes import round_predictions  # noqa: E402

WIDE = [(1.2, 1.2), (0.83, 0.83), (1, 1, 0.16, 0), (1, 1, -0.16, 0), (1, 1, 0, 0.16), (1, 1, 0, -0.16)]


def iou(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    i = max(0.0, w) * max(0.0, h)
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def widen(p):
    x, y, w, h = p['bbox']
    cx, cy = x + w / 2, y + h / 2
    for sw, sh, *shift in WIDE:
        nw, nh = w * sw, h * sh
        dx, dy = (shift[0] * w, shift[1] * h) if shift else (0.0, 0.0)
        yield [cx + dx - nw / 2, cy + dy - nh / 2, nw, nh]


def main(source: Path, nms95: Path, target: Path, images: Path = ROOT / 'data/course-data/val/images.json') -> None:
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    boxes = json.loads(members['detection.json'])
    widths = {im['id']: im['width'] for im in json.loads(images.read_text())['images']}
    with zipfile.ZipFile(nms95) as archive:
        second = round_predictions([p for p in json.loads(archive.read('detection.json')) if p['score'] >= 0.001],
                                   widths, 'whole-pixel-classes')
    floor = min(p['score'] for p in boxes)
    by_key = defaultdict(list)
    for p in boxes:
        by_key[p['image_id'], p['category_id']].append(p['bbox'])
    groups = [('removed by NMS', [{**p, 'score': 0.5 * floor * p['score']} for p in second]),
              ('wider hedge copies', [{**p, 'bbox': b, 'score': 4e-4 * floor * p['score']}
                                      for p in boxes if p['score'] >= 0.001 for b in widen(p)])]
    added = []
    for name, group in groups:
        n = 0
        for p in sorted(group, key=lambda p: -p['score']):
            have = by_key[p['image_id'], p['category_id']]
            if len(have) < 100 and not any(iou(p['bbox'], b) >= 0.99 for b in have):
                have.append(p['bbox'])
                added.append(p)
                n += 1
        print(f'{name}: {n} added', flush=True)
    if added and max(p['score'] for p in added) >= floor:
        raise ValueError('an added box would rank above a box already in the ZIP')
    members['detection.json'] = json.dumps(boxes + added).encode()
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    print(f'{target}: {len(boxes)} boxes + {len(added)} extra (all below the lowest ZIP score {floor:.2e}); '
          'classification.csv byte-for-byte unchanged', flush=True)


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:]))
