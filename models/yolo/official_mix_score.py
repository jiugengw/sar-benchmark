"""Clean-holdout mAP reweighted to the official validation set's mix of image sources, for choosing submissions.

The official set differs from our holdout in where each class's objects come from: e.g. 65% of its oil-tank images are
256 px wide (holdout 38%) and 75% of its harbor images (holdout 29%); our model scores 256 px oil tanks about three
times higher than 512 px ones. mAP averages classes, so within each class what matters is the source mix. This
scores each class's AP separately on the clean holdout's 256 px and 512 px images and averages them with the
official set's shares (from a submission ZIP's predicted image classes, about 94% accurate, and the image widths).
A per-source AP on few images is noisy, so read differences under about 0.5 as ties.

    python models/yolo/official_mix_score.py results/memorise-lr2-332326/one2many results/memorise-lr4-332325/one2many
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))

NAMES = ('aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil_tank', 'playground', 'ship', 'wind_turbine')


def official_shares(zip_path: Path, images_json: Path) -> dict[str, float]:
    """Per class: share of the official images of that (predicted) class that are 256 px wide."""
    width = {im['id']: im['width'] for im in json.loads(images_json.read_text())['images']}
    with zipfile.ZipFile(zip_path) as archive:
        rows = csv.DictReader(io.StringIO(archive.read('classification.csv').decode()))
        predicted = {int(r['image_id']): int(r['label']) for r in rows}
    count = Counter((predicted[i], width[i] == 256) for i in predicted)
    return {name: count[(k, True)] / max(1, count[(k, True)] + count[(k, False)]) for k, name in enumerate(NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', type=Path, nargs='+', help='folders with predictions_val.json')
    parser.add_argument('--zip', type=Path, default=ROOT / 'submissions/memorise-lr2-332326/memorise-lr2-332326.zip',
                        help='a submission ZIP whose predicted image classes give the official class mix')
    parser.add_argument('--images', type=Path, default=ROOT / 'data/course-data/val/images.json')
    parser.add_argument('--round', action='store_true', help='also score each with whole-pixel rounding')
    args = parser.parse_args()
    from resolution_probe import HOLDOUT, coco
    from round_boxes import round_predictions
    from sarbench.twins import clean_image_ids

    shares = official_shares(args.zip, args.images)
    gt = json.loads(HOLDOUT.read_text())
    clean = set(clean_image_ids(gt['images']))
    width = {im['id']: im['width'] for im in gt['images']}
    small, large = {i for i in clean if width[i] == 256}, {i for i in clean if width[i] != 256}
    print('official share of 256 px images per class: ' + ', '.join(f'{k} {100 * v:.0f}%' for k, v in shares.items()))
    print(f"{'predictions':58s} {'clean mAP':>9s} {'official-mix mAP':>16s}")
    for folder in args.results:
        predictions = json.loads((folder / 'predictions_val.json').read_text())
        variants = [('', predictions)]
        if args.round:
            variants.append((' + rounding', round_predictions(predictions, width, 'whole-pixel-classes')))
        for suffix, boxes in variants:
            m, _ = coco(HOLDOUT, boxes, clean)
            _, ap_small = coco(HOLDOUT, boxes, small)
            _, ap_large = coco(HOLDOUT, boxes, large)
            mixed = []
            for name in NAMES:
                s, l = ap_small.get(name, float('nan')), ap_large.get(name, float('nan'))
                if s != s:  # no clean 256 px images of this class
                    mixed.append(l)
                elif l != l:
                    mixed.append(s)
                else:
                    mixed.append(shares[name] * s + (1 - shares[name]) * l)
            print(f"{str(folder)[-50:] + suffix:58s} {100 * m:9.2f} {100 * sum(mixed) / len(mixed):16.2f}", flush=True)


if __name__ == '__main__':
    main()
