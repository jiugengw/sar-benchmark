#!/usr/bin/env python3
"""Rewrite a submission ZIP's classification.csv with the two-head image classifier; detections are untouched.

p = w * p_conv + (1 - w) * p_mean, where p_mean is the ZIP's own classifier (make_submission.py saves its
probabilities beside the ZIP) and p_conv the convolutional branch (conv_head_predict.py). Then the size rule:
classes never seen at the image's original size in the training data get probability 0 (256 px images hold only
bridge, ship, harbour, oil tank or aircraft; 224 px only aircraft; any other size only ships). w = 0.7 was chosen on
the holdout (chosen on either half, it scored on the other: macro-F1 95.46 -> 95.83).

    python models/yolo/combine_classification.py --zip in.zip --mean-probs in_base.probabilities.json \\
        --conv-probs val_probs.json --out out.zip
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--mean-probs', type=Path, required=True)
    parser.add_argument('--conv-probs', type=Path, required=True)
    parser.add_argument('--images', type=Path, default=ROOT / 'data/course-data/val/images.json')
    parser.add_argument('--weight', type=float, default=0.7)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from classifier_combo_study import apply_size_rule, size_allowed

    images = json.loads(args.images.read_text())['images']
    mean, conv = json.loads(args.mean_probs.read_text()), json.loads(args.conv_probs.read_text())
    ids = [im['id'] for im in images]
    missing = [i for i in ids if str(i) not in mean or str(i) not in conv]
    if missing:
        raise ValueError(f'{len(missing)} images lack probabilities, e.g. {missing[:3]}')
    p = args.weight * np.array([conv[str(i)] for i in ids]) + (1 - args.weight) * np.array([mean[str(i)] for i in ids])
    p = apply_size_rule(p, [(im['width'], im['height']) for im in images], size_allowed())
    labels = p.argmax(1)
    with zipfile.ZipFile(args.zip) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    old = dict(line.split(',') for line in members['classification.csv'].decode().strip().splitlines()[1:])
    members['classification.csv'] = ('image_id,label\n' + ''.join(f'{i},{int(k)}\n' for i, k in zip(ids, labels))).encode()
    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    changed = sum(1 for i, k in zip(ids, labels) if old.get(str(i)) != str(int(k)))
    print(f'{args.out}: {len(ids)} image labels, {changed} changed from the ZIP\'s own classifier; detections unchanged')


if __name__ == '__main__':
    main()
