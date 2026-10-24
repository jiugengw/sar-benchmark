"""Holdout class probabilities of a saved image-classifier head on its detector (for combining classifiers).

    python models/yolo/head_probs.py --detector runs/<run>/train/weights/best.pt \\
        --classifier runs/<pkg>/image_classifier.pt --out runs/<pkg>/holdout_probs.json [--device cpu]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--classifier', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    from detector_support import load_detector
    from image_head import load_classifier_head
    from train_image_head import ImageLabels, extract_features

    device = torch.device(args.device)
    dataset_root, data = ROOT / 'data/yolo11s-sar-512', ROOT / 'data/course-data'
    split = json.loads((ROOT / 'runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json').read_text())
    with (data / 'train' / 'labels.csv').open(newline='') as stream:
        labels = {int(r['image_id']): int(r['label']) for r in csv.DictReader(stream)}
    val_ids = list(map(int, split['val_image_ids']))
    loader = DataLoader(ImageLabels(val_ids, labels, json.loads((dataset_root / 'image_map.json').read_text()),
                                    dataset_root), batch_size=16, shuffle=False, num_workers=4)
    detector = load_detector(args.detector, 'yolo')
    detector.model.to(device).eval()
    head, payload = load_classifier_head(args.classifier, device)
    features, _ = extract_features(detector, loader, device, 'yolo', payload.get('feature_pooling', 'mean'))
    with torch.no_grad():
        probs = torch.softmax(head(features.to(device)), 1).cpu().tolist()
    args.out.write_text(json.dumps({str(i): p for i, p in zip(val_ids, probs)}))
    print(f'HEAD_PROBS {len(val_ids)} images -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
