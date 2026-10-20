"""Clean the TRAINING labels with out-of-fold predictions (the holdout and all official images stay untouched).

A model cannot judge labels it was trained on (it fits them), so each training image is judged by a model that never
saw it: two 100-epoch models, each trained on one seeded half of the training split (build_variant_dataset.py
--train-fraction 0.5, with and without --complement), each predicts the other half.

Rules, conservative on purpose:
- remove a label when the out-of-fold model has no box of that class overlapping it (IoU >= --keep-iou) at any score
  >= --keep-score: an object a good detector cannot see at all is most likely invisible in the radar image;
- add a box when the out-of-fold model is confident (score >= --add-score), the class is the image's own class (every
  training image holds exactly one class) and no label of any class overlaps it (IoU >= --add-iou): a missing label;
- box edges are left alone (the hidden-convention study found their disagreement unpredictable).

Writes a cleaned label folder, an image list and a dataset YAML (images are reached through a folder symlink and a
list file, so no image is copied: the cluster file limit), plus relabel_report.json with counts per class and source.

    python models/yolo/relabel_train.py --model-a runs/<half A>/train/weights/best.pt \\
        --model-b runs/<half B>/train/weights/best.pt --out data/yolo-sar-512-relabel
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))
SOURCE = ROOT / 'data/yolo11s-sar-512'
CLASSES = ('aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil_tank', 'playground', 'ship', 'wind_turbine')


def read_labels(path: Path) -> np.ndarray:
    """YOLO label file -> (n, 5) class, cx, cy, w, h (normalised)."""
    rows = [list(map(float, line.split()[:5])) for line in path.read_text().splitlines() if line.strip()]
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def xyxy(cxcywh: np.ndarray) -> np.ndarray:
    return np.concatenate([cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2], 1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a, area_b = (a[:, 2:] - a[:, :2]).prod(-1), (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def predict(weights: Path, files: list[Path], device='0') -> dict[str, np.ndarray]:
    """file name -> (n, 6) class, score, cx, cy, w, h (normalised), one-to-many head + NMS 0.7, as submitted."""
    from detector_support import load_detector

    detector = load_detector(weights, 'yolo')
    detector.model.end2end = False
    out = {}
    for start in range(0, len(files), 64):
        for result in detector.predict(source=[str(f) for f in files[start:start + 64]], imgsz=512, batch=16, device=device,
                                       conf=0.001, iou=0.7, nms=True, max_det=300, verbose=False, save=False):
            b = result.boxes
            out[Path(result.path).name] = np.concatenate(
                [b.cls.cpu().numpy()[:, None], b.conf.cpu().numpy()[:, None], b.xywhn.cpu().numpy()], 1) \
                if len(b) else np.zeros((0, 6))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-a', type=Path, required=True, help='trained on the --train-fraction 0.5 half')
    parser.add_argument('--model-b', type=Path, required=True, help='trained on the complementary half')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--keep-score', type=float, default=0.01)
    parser.add_argument('--keep-iou', type=float, default=0.3)
    parser.add_argument('--add-score', type=float, default=0.6)
    parser.add_argument('--add-iou', type=float, default=0.2)
    parser.add_argument('--device', default='0')
    parser.add_argument('--limit', type=int, help='only the first N training images (to test the script)')
    args = parser.parse_args()
    from build_variant_dataset import kept_train_files

    image_map = json.loads((SOURCE / 'image_map.json').read_text())
    half_a = kept_train_files(image_map, 0.5)
    train_files = sorted((SOURCE / 'images/train').glob('*.png'))[: args.limit]
    in_a = [f for f in train_files if f.name in half_a]
    in_b = [f for f in train_files if f.name not in half_a]
    predictions = predict(args.model_b, in_a, args.device) | predict(args.model_a, in_b, args.device)  # each image judged by the other half's model
    width = {Path(k).name: v['width'] for k, v in image_map.items()}

    out = args.out.resolve()
    (out / 'labels/train').mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    os.symlink(SOURCE.resolve() / 'images/train', out / 'images/train', target_is_directory=True)
    removed, added, kept_count = Counter(), Counter(), Counter()
    for f in train_files:
        labels = read_labels(SOURCE / 'labels/train' / f'{f.stem}.txt')
        pred = predictions.get(f.name, np.zeros((0, 6)))
        source = 'src256' if width.get(f.name) == 256 else 'src512'
        keep = np.ones(len(labels), bool)
        for i, (c, *box) in enumerate(labels):
            same = pred[(pred[:, 0] == c) & (pred[:, 1] >= args.keep_score)]
            if iou_matrix(xyxy(np.array([box])), xyxy(same[:, 2:])).max(initial=0) < args.keep_iou:
                keep[i] = False
                removed[(CLASSES[int(c)], source)] += 1
            else:
                kept_count[(CLASSES[int(c)], source)] += 1
        rows = [labels[i] for i in range(len(labels)) if keep[i]]
        image_class = Counter(labels[:, 0].astype(int)).most_common(1)[0][0] if len(labels) else None
        confident = pred[(pred[:, 1] >= args.add_score)]
        if image_class is not None and len(confident):
            confident = confident[confident[:, 0] == image_class]
            overlap = iou_matrix(xyxy(confident[:, 2:]), xyxy(labels[:, 1:])).max(1, initial=0) if len(labels) \
                else np.zeros(len(confident))
            for p in confident[overlap < args.add_iou]:
                rows.append(np.array([p[0], *p[2:]]))
                added[(CLASSES[int(p[0])], source)] += 1
        (out / 'labels/train' / f'{f.stem}.txt').write_text(
            ''.join(f'{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n' for r in rows))
    (out / 'train.txt').write_text(''.join(f'{out / "images/train" / f.name}\n' for f in train_files))
    names = ''.join(f'\n  {i}: {n}' for i, n in enumerate(CLASSES))
    (out / 'dataset.yaml').write_text(f'path: {out}\ntrain: {out / "train.txt"}\nval: {SOURCE.resolve() / "images/val"}\n'
                                      f'nc: {len(CLASSES)}\nnames:{names}\n')
    report = {f'{c} {s}': {'kept': kept_count[(c, s)], 'removed': removed[(c, s)], 'added': added[(c, s)]}
              for c in CLASSES for s in ('src512', 'src256') if kept_count[(c, s)] + removed[(c, s)] + added[(c, s)]}
    (out / 'relabel_report.json').write_text(json.dumps({'settings': vars(args) | {'model_a': str(args.model_a),
                                                         'model_b': str(args.model_b), 'out': str(out)},
                                                         'per_class_source': report}, indent=2) + '\n')
    print(f"{'class / source':22s} {'kept':>6s} {'removed':>8s} {'added':>6s}")
    for key, v in report.items():
        print(f"{key:22s} {v['kept']:6d} {v['removed']:8d} {v['added']:6d}")
    print(f'RELABEL_READY {out / "dataset.yaml"}')


if __name__ == '__main__':
    main()
