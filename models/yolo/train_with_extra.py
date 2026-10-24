#!/usr/bin/env python3
"""finetune_detector.py with extra labelled training images read from ONE zip file (the cluster's file quota).

The extra images are added to the training set only (the holdout is untouched). Each extra label is an Ultralytics
label dict whose im_file is 'zip://<member>', e.g. the synthetic polarisation views from pol_translate.py.

    python models/yolo/train_with_extra.py --extra-zip runs/pol-views/views.zip \\
        --extra-labels runs/pol-views/views_labels.pkl [finetune_detector.py options...]
"""
from __future__ import annotations

import os
import pickle
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))


def install(zip_path: Path, labels_path: Path) -> None:
    from ultralytics.data import base as ul_base
    from ultralytics.data import dataset as ul_dataset

    handles = {}

    def handle():
        pid = os.getpid()  # one handle per data-loader worker
        if pid not in handles:
            handles[pid] = zipfile.ZipFile(zip_path)
        return handles[pid]

    original_imread = ul_base.imread

    def imread(filename, flags=cv2.IMREAD_COLOR):
        name = str(filename)
        if name.startswith('zip:/'):
            data = np.frombuffer(handle().read(name.split('zip:/', 1)[1].lstrip('/')), np.uint8)
            return cv2.imdecode(data, flags)
        return original_imread(filename, flags)

    ul_base.imread = imread
    with open(labels_path, 'rb') as fh:
        extra = pickle.load(fh)
    original_get_labels = ul_dataset.YOLODataset.get_labels

    def get_labels(self):
        labels = original_get_labels(self)
        if self.augment:  # training set only
            labels = labels + [dict(lb, cls=lb['cls'].copy(), bboxes=lb['bboxes'].copy()) for lb in extra]
            self.im_files = [lb['im_file'] for lb in labels]
            print(f'extra training images: {len(extra)} added ({len(labels)} in all)', flush=True)
        return labels

    ul_dataset.YOLODataset.get_labels = get_labels


def main() -> None:
    argv = sys.argv[1:]
    values = {}
    for flag in ('--extra-zip', '--extra-labels'):
        k = argv.index(flag)
        values[flag] = Path(argv[k + 1])
        del argv[k:k + 2]
    install(values['--extra-zip'], values['--extra-labels'])
    import finetune_detector
    sys.argv = ['finetune_detector.py'] + argv
    finetune_detector.main()


if __name__ == '__main__':
    main()
