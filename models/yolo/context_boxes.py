"""Shrink predicted context boxes back to object size and score the result (experiment A).

A detector trained on build_variant_dataset.py --context-classes ... --context-factor K predicts boxes K times
larger than the objects of those classes. This shrinks those classes' predicted boxes by K around their centre, writes
the corrected predictions next to the originals, and scores full and clean holdout mAP with per-class AP.

    python models/yolo/context_boxes.py results/ft40-ctx3-123/one2many --classes bridge --factor 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def shrink(predictions: list[dict], category_ids: set[int], factor: float) -> list[dict]:
    """COCO-format predictions; boxes of `category_ids` shrink by `factor` around their centre."""
    out = []
    for p in predictions:
        if p['category_id'] in category_ids:
            x, y, w, h = p['bbox']
            cx, cy = x + w / 2, y + h / 2
            w, h = w / factor, h / factor
            p = dict(p, bbox=[cx - w / 2, cy - h / 2, w, h])
        out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', type=Path, help='folder with predictions_val.json and instances_val_holdout.json')
    parser.add_argument('--classes', required=True, help='comma-separated class names trained with context boxes')
    parser.add_argument('--factor', type=float, required=True)
    args = parser.parse_args()
    from sarbench.metrics import detection_metrics
    from sarbench.twins import clean_image_ids, score

    gt_path = args.results / 'instances_val_holdout.json'
    gt = json.loads(gt_path.read_text())
    by_name = {c['name']: c['id'] for c in gt['categories']}
    wanted = [n.strip() for n in args.classes.split(',') if n.strip()]
    if set(wanted) - set(by_name):
        raise ValueError(f'unknown classes {set(wanted) - set(by_name)}')
    out = args.results.parent / f'{args.results.name}-shrunk'
    out.mkdir(exist_ok=True)
    predictions = shrink(json.loads((args.results / 'predictions_val.json').read_text()),
                         {by_name[n] for n in wanted}, args.factor)
    (out / 'predictions_val.json').write_text(json.dumps(predictions) + '\n')
    for name in ('instances_val_holdout.json', 'image_classification_predictions.json'):
        if (args.results / name).is_file():
            shutil.copy2(args.results / name, out / name)
    print(f'== {out}: {wanted} boxes shrunk by {args.factor}x', flush=True)
    report = score(out, ROOT / 'data/course-data')
    images = gt['images']
    for subset, ids in (('full', [im['id'] for im in images]), ('clean', clean_image_ids(images))):
        per_class = detection_metrics(str(gt_path), predictions, ids)['per_class_AP']
        report[subset]['per_class_AP'] = per_class
        print(f'{subset:5s} per-class AP: ' + ', '.join(f'{n} {v * 100:.1f}' for n, v in per_class.items()), flush=True)
    (out / 'full_clean_scores.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
