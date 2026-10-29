#!/usr/bin/env python3
"""A convolutional image-classification branch on the frozen detector's feature maps (still one model).

The current classifier averages every feature map over the whole image before deciding, so a few 10 px playgrounds
or bridges barely move it, and most of its mistakes are bridge / playground / oil tank / harbour scenes. This branch
keeps the maps: per detection level (P2 ... P5) a 1x1 projection, two 3x3 conv blocks at that level's resolution,
then max AND mean pooling (max keeps a single strong object response), concatenated across levels -> MLP -> 9
classes. The detector stays frozen and runs on every batch; training images are flipped at random.

Reported on the 939-image holdout: macro-F1 at the best epoch (optimistic: the holdout picks it) and cross-fitted
(the epoch picked on one half of the holdout, twin groups together, scored on the other half), full and clean.

    python models/yolo/conv_head_study.py --detector runs/ft40-m4rep-bp4-memorise-335138/train/weights/best.pt \\
        --output runs/conv-head-bp4/conv_head.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))


class ConvBranch(nn.Module):
    def __init__(self, channels: list[int], width: int = 128, hidden: int = 512, classes: int = 9,
                 dropout: float = 0.3):
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.SiLU())
        self.levels = nn.ModuleList(nn.Sequential(nn.Conv2d(c, width, 1, bias=False), nn.BatchNorm2d(width),
                                                  nn.SiLU(), block(width, width), block(width, width))
                                    for c in channels)
        self.mlp = nn.Sequential(nn.LayerNorm(2 * width * len(channels)), nn.Dropout(dropout),
                                 nn.Linear(2 * width * len(channels), hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, classes))

    def forward(self, maps):
        pooled = []
        for level, feature in zip(self.levels, maps):
            x = level(feature.float())
            pooled += [x.amax(dim=(2, 3)), x.mean(dim=(2, 3))]
        return self.mlp(torch.cat(pooled, dim=1))


def macro_f1(truth, pred, classes=9):
    f1s = []
    for c in range(classes):
        tp = sum(1 for t, p in zip(truth, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(truth, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(truth, pred) if t == c and p != c)
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return 100 * float(np.mean(f1s))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--all-data', action='store_true',
                        help='also train on the holdout and keep the last epoch (for an all-data detector; the holdout '
                             'F1 printed is then optimistic)')
    parser.add_argument('--views', choices=('flips', 'unseen'), default='flips',
                        help="training views: 'flips' (random flips) or 'unseen' (quarter turns + "
                             "brightness only: views a flip-memorised detector never trained on; see "
                             "train_image_head.unseen_view)")
    parser.add_argument('--swa-from', type=int, default=0,
                        help='also weight-average the epochs from this one to the last (BatchNorm recomputed), score '
                             'it on the holdout, and save it instead of a single epoch')
    parser.add_argument('--holdout-gt', type=Path,
                        default=ROOT / 'results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json')
    args = parser.parse_args()
    from detector_support import detection_head, feature_maps_from_hook, load_detector
    from ranking_study import twin_groups
    from sarbench.twins import clean_image_ids
    from train_image_head import ImageLabels, unseen_view

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device('cuda:0')
    data, dataset_root = ROOT / 'data/course-data', ROOT / 'data/yolo11s-sar-512'
    split = json.loads((ROOT / 'runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json').read_text())
    with (data / 'train' / 'labels.csv').open(newline='') as stream:
        labels = {int(r['image_id']): int(r['label']) for r in csv.DictReader(stream)}
    image_map = json.loads((dataset_root / 'image_map.json').read_text())
    train_ids, val_ids = list(map(int, split['train_image_ids'])), list(map(int, split['val_image_ids']))
    if args.all_data:
        train_ids = train_ids + val_ids
    train_loader = DataLoader(ImageLabels(train_ids, labels, image_map, dataset_root), batch_size=args.batch,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ImageLabels(val_ids, labels, image_map, dataset_root), batch_size=args.batch,
                            shuffle=False, num_workers=4, pin_memory=True)

    detector = load_detector(args.detector, 'yolo')
    model = detector.model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    captured = []
    detection_head(model).register_forward_hook(
        lambda _m, inputs, output: captured.append([f.detach() for f in feature_maps_from_hook('yolo', inputs, output)]))

    def maps(images):
        captured.clear()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            model(images)
        return captured.pop()

    with torch.no_grad():
        channels = [f.shape[1] for f in maps(torch.zeros(1, 3, 512, 512, device=device))]
    branch = ConvBranch(channels, width=args.width).to(device)
    n_params = sum(p.numel() for p in branch.parameters())
    detector_params = sum(p.numel() for p in model.parameters())
    print(f'feature levels {channels}; branch {n_params:,} parameters; detector {detector_params:,}', flush=True)
    if n_params + detector_params > 40_000_000:
        raise RuntimeError('over the 40M trainable-parameter budget')
    counts = torch.bincount(torch.tensor([labels[i] for i in train_ids]), minlength=9).float()
    criterion = nn.CrossEntropyLoss(weight=(counts.sum() / (9 * counts.clamp_min(1))).to(device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(branch.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    gt = json.loads(args.holdout_gt.read_text())
    clean = set(clean_image_ids(gt['images']))
    groups = twin_groups(gt['images'])
    unique = sorted(set(groups.values()))
    half = set(np.random.default_rng(0).permutation(unique)[: len(unique) // 2])
    in_a = [groups[i] in half for i in val_ids]
    truth = [labels[i] for i in val_ids]
    history, probs_by_epoch, states = [], [], []
    for epoch in range(1, args.epochs + 1):
        branch.train()
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device)
            if args.views == 'unseen':
                images = unseen_view(images)
            else:
                flip = [d for d, keep in ((-1, random.random() < 0.5), (-2, random.random() < 0.5)) if keep]
                if flip:
                    images = images.flip(flip)
            loss = criterion(branch(maps(images)), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        branch.eval()
        probs = []
        with torch.no_grad():
            for images, _ in val_loader:
                probs.append(torch.softmax(branch(maps(images.to(device))), 1).cpu())
        probs = torch.cat(probs).numpy()
        pred = probs.argmax(1).tolist()
        f_all = macro_f1(truth, pred)
        f_clean = macro_f1([t for t, i in zip(truth, val_ids) if i in clean], [p for p, i in zip(pred, val_ids) if i in clean])
        f_a = macro_f1([t for t, a in zip(truth, in_a) if a], [p for p, a in zip(pred, in_a) if a])
        f_b = macro_f1([t for t, a in zip(truth, in_a) if not a], [p for p, a in zip(pred, in_a) if not a])
        history.append((epoch, f_all, f_clean, f_a, f_b))
        probs_by_epoch.append(probs)
        states.append({k: v.detach().cpu().clone() for k, v in branch.state_dict().items()})
        print(f'epoch {epoch}: holdout macro-F1 {f_all:.2f} (clean {f_clean:.2f}; halves {f_a:.2f} / {f_b:.2f})', flush=True)
    best = max(history, key=lambda h: h[1])
    pick_a = max(history, key=lambda h: h[3])
    pick_b = max(history, key=lambda h: h[4])
    print(f'best epoch {best[0]}: F1 {best[1]:.2f} full, {best[2]:.2f} clean (optimistic: picked on the holdout)')
    print(f'cross-fitted: epoch picked on half A ({pick_a[0]}) scores {pick_a[4]:.2f} on half B; '
          f'picked on half B ({pick_b[0]}) scores {pick_b[3]:.2f} on half A; mean {(pick_a[4] + pick_b[3]) / 2:.2f}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chosen = args.epochs if args.all_data else best[0]  # all data: the holdout was trained on, keep the last epoch
    state, probs_out = states[chosen - 1], probs_by_epoch[chosen - 1]
    if args.swa_from:
        from branch_head_study import average_states, recompute_batchnorm
        branch.load_state_dict(average_states(states, args.swa_from))
        recompute_batchnorm(branch, (maps(images.to(device)) for images, _ in train_loader))
        probs = []
        with torch.no_grad():
            for images, _ in val_loader:
                probs.append(torch.softmax(branch(maps(images.to(device))), 1).cpu())
        probs_out = torch.cat(probs).numpy(); pred = probs_out.argmax(1).tolist()
        print(f'SWA epochs {args.swa_from}-{args.epochs}: holdout macro-F1 {macro_f1(truth, pred):.2f} (clean '
              f'{macro_f1([t for t, i in zip(truth, val_ids) if i in clean], [p for p, i in zip(pred, val_ids) if i in clean]):.2f}; '
              f'halves {macro_f1([t for t, a in zip(truth, in_a) if a], [p for p, a in zip(pred, in_a) if a]):.2f} / '
              f'{macro_f1([t for t, a in zip(truth, in_a) if not a], [p for p, a in zip(pred, in_a) if not a]):.2f}) '
              f'against the last epoch {history[-1][1]:.2f}', flush=True)
        state, chosen = {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()}, f'swa{args.swa_from}-{args.epochs}'
    np.save(args.output.with_suffix('.holdout_probs.npy'), probs_out)
    np.save(args.output.with_suffix('.holdout_probs_all_epochs.npy'), np.stack(probs_by_epoch))
    torch.save({'format': 'sar-conv-branch-v1', 'channels': channels, 'width': args.width, 'epoch': chosen, 'views': args.views,
                'all_data': args.all_data, 'detector': str(args.detector), 'state_dict': state}, args.output)
    print(f'saved the epoch-{chosen} branch to {args.output}', flush=True)
    (args.output.with_suffix('.json')).write_text(json.dumps({'history': history, 'val_ids': val_ids}))


if __name__ == '__main__':
    main()
