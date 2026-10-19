"""Which final step turns the detector's raw boxes into the best mAP? Tested on the holdout, no retraining.

1. Predict the holdout once through the exact submission code with NMS almost off (IoU 0.95), so the overlapping
   boxes the detector makes for each object are kept (GPU, a few minutes).
2. Offline, apply each final step to those same boxes and score full and clean holdout mAP (CPU):
     nms(t)            standard class-wise NMS at overlap t (our submissions use 0.7)
     fuse(t)           weighted box fusion: overlapping boxes (IoU > t) merge into one box at their score-weighted
                       average position, with the best score of the group
     nms+avg(t, a)     NMS at t, then each kept box moves to the score-weighted average of all boxes overlapping it by
                       more than a (it keeps its own score)
   Fusion averages several noisy estimates of the same edges, which is what the strict 0.80-0.95 thresholds reward.
   All of them are still one model and one forward pass.

    python models/yolo/postprocess_study.py --tag pp-memorise-lr2 \\
        --weights runs/ft40-memorise-lr2-332326/train/weights/best.pt \\
        --classifier runs/memorise-lr2-332326-package/image_classifier.pt
    python models/yolo/postprocess_study.py --tag pp-memorise-lr2 --offline   # step 2 again on saved boxes
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy boxes a (n, 4) and b (m, 4)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = list(np.argsort(-scores))
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        if order:
            overlap = iou_matrix(boxes[[i]], boxes[order])[0]
            order = [j for j, v in zip(order, overlap) if v <= threshold]
    return keep


def fuse(boxes: np.ndarray, scores: np.ndarray, threshold: float):
    """Greedy weighted box fusion of one class in one image: (fused boxes, fused scores)."""
    clusters: list[list[int]] = []
    fused: list[np.ndarray] = []
    for i in np.argsort(-scores):
        if fused:
            overlap = iou_matrix(boxes[[i]], np.stack(fused))[0]
            j = int(np.argmax(overlap))
            if overlap[j] > threshold:
                clusters[j].append(i)
                w = scores[clusters[j]]
                fused[j] = (boxes[clusters[j]] * w[:, None]).sum(0) / w.sum()
                continue
        clusters.append([i])
        fused.append(boxes[i].copy())
    return np.stack(fused), np.array([scores[c].max() for c in clusters])


def nms_average(boxes: np.ndarray, scores: np.ndarray, threshold: float, average: float):
    keep = nms(boxes, scores, threshold)
    overlap = iou_matrix(boxes[keep], boxes)
    out = []
    for row, i in zip(overlap, keep):
        group = row > average
        w = scores[group]
        out.append((boxes[group] * w[:, None]).sum(0) / w.sum())
    return np.stack(out), scores[keep]


def apply(predictions: list[dict], method: str, *params) -> list[dict]:
    groups = defaultdict(list)
    for p in predictions:
        groups[(p['image_id'], p['category_id'])].append(p)
    out = []
    for (image_id, category_id), items in groups.items():
        xywh = np.array([p['bbox'] for p in items], dtype=float)
        boxes = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], 1)
        scores = np.array([p['score'] for p in items], dtype=float)
        if method == 'nms':
            keep = nms(boxes, scores, *params)
            boxes, scores = boxes[keep], scores[keep]
        elif method == 'fuse':
            boxes, scores = fuse(boxes, scores, *params)
        elif method == 'nms+avg':
            boxes, scores = nms_average(boxes, scores, *params)
        for b, s in zip(boxes, scores):
            out.append({'image_id': image_id, 'category_id': category_id,
                        'bbox': [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], 'score': float(s)})
    # the submission keeps at most 300 boxes per image, highest scores first
    by_image = defaultdict(list)
    for p in out:
        by_image[p['image_id']].append(p)
    return [p for items in by_image.values() for p in sorted(items, key=lambda p: -p['score'])[:300]]


def study(raw: Path) -> list[dict]:
    from sarbench.metrics import detection_metrics
    from sarbench.twins import clean_image_ids

    gt_path = raw / 'instances_val_holdout.json'
    images = json.loads(gt_path.read_text())['images']
    full, clean = [im['id'] for im in images], clean_image_ids(images)
    predictions = json.loads((raw / 'predictions_val.json').read_text())
    variants = [('nms', 0.5), ('nms', 0.6), ('nms', 0.7), ('nms', 0.8),
                ('fuse', 0.5), ('fuse', 0.6), ('fuse', 0.7), ('fuse', 0.8),
                ('nms+avg', 0.7, 0.6), ('nms+avg', 0.7, 0.75), ('nms+avg', 0.6, 0.6)]
    rows = []
    print(f'{len(predictions)} raw boxes (NMS 0.95)', flush=True)
    for method, *params in variants:
        boxes = apply(predictions, method, *params)
        m_full = detection_metrics(str(gt_path), boxes, full)
        m_clean = detection_metrics(str(gt_path), boxes, clean)
        row = {'method': method, 'params': params, 'boxes': len(boxes), 'full_mAP': m_full['mAP'],
               'clean_mAP': m_clean['mAP'], 'clean_AP75': m_clean['AP75'], 'clean_per_class_AP': m_clean['per_class_AP']}
        rows.append(row)
        print(f"{method:8s} {str(params):12s} boxes {len(boxes):6d}  full mAP {100 * m_full['mAP']:6.2f}  "
              f"clean mAP {100 * m_clean['mAP']:6.2f}  clean AP75 {100 * m_clean['AP75']:6.2f}", flush=True)
    (raw.parent / 'postprocess_study.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True, help='results/<tag>/raw holds the near-NMS-free holdout predictions')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--classifier', type=Path)
    parser.add_argument('--offline', action='store_true', help='skip prediction; rerun the comparison on saved boxes')
    args = parser.parse_args()
    raw = ROOT / 'results' / args.tag / 'raw'
    if not args.offline:
        from models.yolo.deadline_recovery import evaluate_export

        evaluate_export(args.weights, args.classifier, 'one2many', raw, nms_iou=0.95)
    study(raw)


if __name__ == '__main__':
    main()
