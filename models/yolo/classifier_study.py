"""Which image-classifier setup gives the best macro-F1 on images it was not tuned on? Holdout only, detector fixed.

The packaged classifier keeps the epoch with the best holdout macro-F1 out of 240, so its holdout F1 is flattered by
that choice, and official F1 has come in 1-2 points lower. Here every variant is scored honestly: the epoch is chosen
on one half of the holdout and scored on the other half (and the other way round, two random splits, averaged).

Variants: training flips on/off (flipped radar images are physically wrong: shadows fall on the wrong side), hidden
width 512/1024/2048, dropout 0.35/0.6. Features are extracted once per flip setting and reused.

    python models/yolo/classifier_study.py --tag cls-memorise-lr2 --detector runs/ft40-memorise-lr2-332326/train/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))

SPLIT = ROOT / 'results/yolo26m-100-p2-328339/seed_0/split.json'
HOLDOUT = ROOT / 'results/yolo26m-100-p2-328339/seed_0/instances_val_holdout.json'
DATASET = ROOT / 'data/yolo11s-sar-512'


def macro_f1(truth, predicted, classes=9) -> float:
    scores = []
    for c in range(classes):
        tp = sum(1 for t, p in zip(truth, predicted) if t == c and p == c)
        fp = sum(1 for t, p in zip(truth, predicted) if t != c and p == c)
        fn = sum(1 for t, p in zip(truth, predicted) if t == c and p != c)
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def honest_f1(per_epoch_predictions, truth, subset) -> float:
    """Pick the epoch on one half of `subset`, score it on the other half; both ways, two random splits, averaged."""
    results = []
    for seed in (0, 1):
        order = subset[:]
        random.Random(seed).shuffle(order)
        halves = order[: len(order) // 2], order[len(order) // 2:]
        for fit, test in (halves, halves[::-1]):
            best = max(range(len(per_epoch_predictions)),
                       key=lambda e: macro_f1([truth[i] for i in fit], [per_epoch_predictions[e][i] for i in fit]))
            results.append(macro_f1([truth[i] for i in test], [per_epoch_predictions[best][i] for i in test]))
    return sum(results) / len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=240)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from detector_support import load_detector
    from image_head import ImageClassifierHead
    from sarbench.twins import clean_image_ids
    from train_image_head import ImageLabels, extract_features

    device = torch.device('cuda:0')
    split = json.loads(SPLIT.read_text())
    with (ROOT / 'data/course-data/train/labels.csv').open(newline='') as stream:
        labels = {int(r['image_id']): int(r['label']) for r in csv.DictReader(stream)}
    image_map = json.loads((DATASET / 'image_map.json').read_text())
    train_ids, val_ids = list(map(int, split['train_image_ids'])), list(map(int, split['val_image_ids']))
    holdout_images = json.loads(HOLDOUT.read_text())['images']
    clean = set(clean_image_ids(holdout_images))
    position = {image_id: k for k, image_id in enumerate(val_ids)}
    subsets = {'full': list(range(len(val_ids))), 'clean': [position[i] for i in val_ids if i in clean]}
    truth = [labels[i] for i in val_ids]

    detector = load_detector(args.detector, 'yolo')
    detector.model.to(device).eval()
    loader = lambda ids: DataLoader(ImageLabels(ids, labels, image_map, DATASET), batch_size=32, num_workers=4)
    val_x, _ = extract_features(detector, loader(val_ids), device, 'yolo', 'spatial', False)
    cache = {flips: extract_features(detector, loader(train_ids), device, 'yolo', 'spatial', flips) for flips in (True, False)}
    del detector
    torch.cuda.empty_cache()
    val_x = val_x.to(device)

    rows = []
    out = ROOT / 'results' / args.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'flips':>5s} {'hidden':>6s} {'dropout':>7s}  {'best-epoch F1 (flattered)':>26s}  {'honest F1 full':>14s}  "
          f"{'honest F1 clean':>15s}  {'last-epoch clean':>16s}", flush=True)
    for flips, hidden, dropout in itertools.product((True, False), (1024, 512, 2048), (0.35, 0.6)):
        torch.manual_seed(0)
        x, y = (t.to(device) for t in cache[flips])
        head = ImageClassifierHead(x.shape[1], 9, dropout=dropout, hidden_dim=hidden).to(device)
        counts = torch.bincount(y, minlength=9).float()
        criterion = torch.nn.CrossEntropyLoss(weight=(counts.sum() / (9 * counts.clamp_min(1))).to(device))
        optimizer = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=1e-3)
        batches = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=True,
                             generator=torch.Generator().manual_seed(0))
        per_epoch = []
        for _ in range(args.epochs):
            head.train()
            for fx, fy in batches:
                optimizer.zero_grad(set_to_none=True)
                criterion(head(fx), fy).backward()
                optimizer.step()
            head.eval()
            with torch.inference_mode():
                per_epoch.append(head(val_x).argmax(1).cpu().tolist())
        flattered = max(macro_f1(truth, p) for p in per_epoch)
        row = {'train_flips': flips, 'hidden_dim': hidden, 'dropout': dropout, 'best_epoch_f1_full': flattered,
               'honest_f1_full': honest_f1(per_epoch, truth, subsets['full']),
               'honest_f1_clean': honest_f1(per_epoch, truth, subsets['clean']),
               'last_epoch_f1_clean': macro_f1([truth[i] for i in subsets['clean']],
                                               [per_epoch[-1][i] for i in subsets['clean']])}
        rows.append(row)
        print(f"{str(flips):>5s} {hidden:6d} {dropout:7.2f}  {100 * flattered:26.2f}  {100 * row['honest_f1_full']:14.2f}  "
              f"{100 * row['honest_f1_clean']:15.2f}  {100 * row['last_epoch_f1_clean']:16.2f}", flush=True)
        (out / 'classifier_study.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
