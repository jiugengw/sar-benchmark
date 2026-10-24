"""Save every candidate box of a detector BEFORE duplicate removal (NMS), for tail studies (Soft-NMS-like ranking).

Same input as the submission writer (grayscale, resized to 512 with bilinear, one-to-many branch, conf 0.001), but
NMS is run with IoU 1.0 so nothing is suppressed, and up to --max-det boxes per image are kept. Boxes are returned
in each image's original pixels, like predictions_val.json.

    python models/yolo/raw_candidates.py --weights runs/<run>/train/weights/best.pt \\
        --images results/<run>/one2many/instances_val_holdout.json --split-root data/course-data/train \\
        --out results/<run>/one2many/raw_candidates.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))
IMAGE_SIZE = 512


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True, help='COCO json whose "images" are predicted')
    parser.add_argument('--split-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-det', type=int, default=3000)
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=1.0, help='NMS IoU; 1.0 keeps every candidate')
    parser.add_argument('--device', default='0')
    parser.add_argument('--native', action='store_true',
                        help='feed images smaller than 512 at their own size instead of upscaling them (a stride-1 '
                             'model then sees a 256 px image on the same grid as a normal model sees it upscaled)')
    args = parser.parse_args()
    from PIL import Image
    from detector_support import load_detector

    images = json.loads(args.images.read_text())['images']
    detector = load_detector(args.weights, 'yolo')
    detector.model.end2end = False  # the one-to-many branch, as submitted
    out = []
    with tempfile.TemporaryDirectory(prefix='raw-candidates-') as temp:
        by_stem, sizes = {}, set()
        for image in images:
            stem = f"{int(image['id']):08d}"
            by_stem[stem] = image
            size = min(image['width'], IMAGE_SIZE) if args.native and image['width'] == image['height'] else IMAGE_SIZE
            sizes.add(size)
            (Path(temp) / str(size)).mkdir(exist_ok=True)
            with Image.open(args.split_root / image['file_name']) as original:
                original.convert('L').resize((size, size), Image.Resampling.BILINEAR).save(
                    Path(temp) / str(size) / f'{stem}.png')
        for size in sorted(sizes):
            for result in detector.predict(source=Path(temp) / str(size), imgsz=size, batch=16, device=args.device,
                                           conf=args.conf, iou=args.iou, nms=True, max_det=args.max_det, stream=True,
                                           verbose=False, save=False):
                image = by_stem[Path(result.path).stem]
                sx, sy = image['width'] / size, image['height'] / size
                for (x1, y1, x2, y2), c, s in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                                  result.boxes.conf.tolist()):
                    x1, x2 = sorted((max(0.0, min(size, x1)), max(0.0, min(size, x2))))
                    y1, y2 = sorted((max(0.0, min(size, y1)), max(0.0, min(size, y2))))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    out.append({'image_id': int(image['id']), 'category_id': int(c) + 1,
                                'bbox': [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy], 'score': float(s)})
    args.out.write_text(json.dumps(out))
    print(f'RAW_CANDIDATES {len(out)} boxes for {len(images)} images -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
