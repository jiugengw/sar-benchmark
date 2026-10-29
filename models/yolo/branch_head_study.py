#!/usr/bin/env python3
"""Image classifier with its own copy of the detector's last backbone stages (still one model; boxes unchanged).

The current classifiers read the frozen detector's final features, which were trained to draw boxes. Most of their
remaining errors are bridge vs playground scenes in 512 px M4-SAR images (22 of 43 clean holdout errors). Here the
frozen detector runs up to backbone layer `--split` (stride 8 by default) and a trainable COPY of the remaining backbone
layers (split .. 10, starting from the trained weights) learns whole-scene classification: max + mean pooling of its
stride-16 and stride-32 outputs -> MLP -> 9 classes. Training images are flipped and rotated by multiples of 90
degrees at random (scene classes do not depend on orientation).

Reported on the 939-image holdout like conv_head_study.py: macro-F1 at the best epoch (optimistic), and cross-fitted
(epoch picked on one twin-grouped half, scored on the other), full and clean.

    python models/yolo/branch_head_study.py --detector runs/soup-bp4/six.pt --output runs/branch-head-six/branch_head.pt
"""
from __future__ import annotations

import argparse
import copy
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


class BranchHead(nn.Module):
    """Trainable copy of backbone layers [split, last] plus a pooled MLP over the last two stage outputs."""

    def __init__(self, layers: nn.ModuleList, channels: list[int], hidden: int = 512, classes: int = 9,
                 dropout: float = 0.3):
        super().__init__()
        self.layers = layers
        self.tap = len(layers) - 5  # output of the stride-16 stage (C3k2 before the last downsampling)
        width = 2 * sum(channels)
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, classes))

    def forward(self, x):
        pooled = []
        for k, layer in enumerate(self.layers):
            x = layer(x)
            if k == self.tap or k == len(self.layers) - 1:
                pooled += [x.float().amax(dim=(2, 3)), x.float().mean(dim=(2, 3))]
        return self.mlp(torch.cat(pooled, dim=1))


def average_states(states: list[dict], start: int) -> dict:
    """Weight average (SWA) of the per-epoch states from epoch `start` (1-based) to the last; integer buffers
    (BatchNorm's batch counter) keep the last value. Still one model; BatchNorm statistics are recomputed after."""
    chosen = states[start - 1:]
    return {k: (torch.stack([s[k].float() for s in chosen]).mean(0).to(v.dtype) if v.is_floating_point() else v)
            for k, v in chosen[-1].items()}


def recompute_batchnorm(module: nn.Module, batches) -> None:
    """Running statistics of every BatchNorm in `module` from one pass over `batches` (inputs ready for module)."""
    norms = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not norms:
        return
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the pass
    module.train()
    with torch.no_grad():
        for x in batches:
            module(x)
    for m, momentum in zip(norms, momenta):
        m.momentum = momentum
    module.eval()


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
    parser.add_argument('--split', type=int, default=5, help='first backbone layer that is copied and trained')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3, help='MLP learning rate; the copied layers get lr / 10')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init-from', type=Path,
                        help='start the copied layers from this checkpoint of the same architecture (e.g. an earlier, '
                             'augmentation-trained stage) instead of the frozen detector')
    parser.add_argument('--all-data', action='store_true',
                        help='also train on the holdout and keep the last epoch (its printed F1 is then optimistic)')
    parser.add_argument('--views', choices=('flips', 'unseen'), default='flips',
                        help="training views: 'flips' (random flips and quarter turns) or 'unseen' (quarter turns + "
                             "brightness only: views a flip-memorised detector never trained on; see "
                             "train_image_head.unseen_view)")
    parser.add_argument('--holdout-gt', type=Path,
                        default=ROOT / 'results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json')
    parser.add_argument('--swa-from', type=int, default=0,
                        help='also weight-average the epochs from this one to the last (BatchNorm recomputed), score '
                             'it on the holdout, and save it instead of a single epoch')
    args = parser.parse_args()
    from detector_support import load_detector
    from ranking_study import twin_groups
    from sarbench.twins import clean_image_ids
    from train_image_head import ImageLabels, unseen_view

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
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

    detector = load_detector(args.detector, 'yolo').model.to(device).eval()
    backbone_end = len(detector.yaml['backbone'])
    if any(detector.model[i].f != -1 for i in range(backbone_end)):
        raise RuntimeError('expected a plain sequential backbone')
    # A TerraMind-fusion detector (terramind_fusion.py): the branch runs the plain detector layers, without TerraMind
    from terramind_fusion import unwrap
    frozen = [unwrap(layer) for layer in detector.model[:args.split]]
    for p in detector.parameters():
        p.requires_grad_(False)
    source = load_detector(args.init_from, 'yolo').model.to(device) if args.init_from else detector
    if source.yaml['backbone'] != detector.yaml['backbone']:
        raise RuntimeError('--init-from must have the same backbone as --detector')
    trunk = nn.ModuleList(copy.deepcopy(unwrap(source.model[i])) for i in range(args.split, backbone_end))
    for p in trunk.parameters():
        p.requires_grad_(True)

    def stem(images):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            x = images
            for layer in frozen:
                x = layer(x)
        return x

    with torch.no_grad():
        x = stem(torch.zeros(1, 3, 512, 512, device=device)).float()
        shapes = []
        for k, layer in enumerate(trunk):
            x = layer(x)
            shapes.append(x.shape[1])
    head = BranchHead(trunk, [shapes[len(trunk) - 5], shapes[-1]]).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    detector_params = sum(p.numel() for p in detector.parameters())
    print(f'copied layers {args.split}-{backbone_end - 1}; branch {n_params:,} parameters; detector {detector_params:,}',
          flush=True)
    if n_params + detector_params > 40_000_000:
        raise RuntimeError('over the 40M trainable-parameter budget')
    counts = torch.bincount(torch.tensor([labels[i] for i in train_ids]), minlength=9).float()
    criterion = nn.CrossEntropyLoss(weight=(counts.sum() / (9 * counts.clamp_min(1))).to(device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW([{'params': head.layers.parameters(), 'lr': args.lr / 10},
                                   {'params': head.mlp.parameters(), 'lr': args.lr}], weight_decay=0.05)
    steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[args.lr / 10, args.lr], total_steps=steps,
                                                    pct_start=0.1)
    scaler = torch.amp.GradScaler()

    gt = json.loads(args.holdout_gt.read_text())
    clean = set(clean_image_ids(gt['images']))
    groups = twin_groups(gt['images'])
    unique = sorted(set(groups.values()))
    half = set(np.random.default_rng(0).permutation(unique)[: len(unique) // 2])
    in_a = [groups[i] in half for i in val_ids]
    truth = [labels[i] for i in val_ids]
    history, probs_by_epoch, states = [], [], []
    for epoch in range(1, args.epochs + 1):
        head.train()
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device)
            if args.views == 'unseen':
                images = unseen_view(images)
            else:
                flip = [d for d, keep in ((-1, random.random() < 0.5), (-2, random.random() < 0.5)) if keep]
                if flip:
                    images = images.flip(flip)
                images = torch.rot90(images, random.randrange(4), dims=(-2, -1))
            with torch.autocast('cuda', dtype=torch.float16):
                loss = criterion(head(stem(images)), targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        head.eval()
        probs = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            for images, _ in val_loader:
                probs.append(torch.softmax(head(stem(images.to(device))).float(), 1).cpu())
        probs = torch.cat(probs).numpy()
        pred = probs.argmax(1).tolist()
        f_all = macro_f1(truth, pred)
        f_clean = macro_f1([t for t, i in zip(truth, val_ids) if i in clean], [p for p, i in zip(pred, val_ids) if i in clean])
        f_a = macro_f1([t for t, a in zip(truth, in_a) if a], [p for p, a in zip(pred, in_a) if a])
        f_b = macro_f1([t for t, a in zip(truth, in_a) if not a], [p for p, a in zip(pred, in_a) if not a])
        history.append((epoch, f_all, f_clean, f_a, f_b))
        probs_by_epoch.append(probs)
        states.append({k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        print(f'epoch {epoch}: holdout macro-F1 {f_all:.2f} (clean {f_clean:.2f}; halves {f_a:.2f} / {f_b:.2f})', flush=True)
    best = max(history, key=lambda h: h[1])
    pick_a = max(history, key=lambda h: h[3])
    pick_b = max(history, key=lambda h: h[4])
    print(f'best epoch {best[0]}: F1 {best[1]:.2f} full, {best[2]:.2f} clean (optimistic: picked on the holdout)')
    print(f'cross-fitted: epoch picked on half A ({pick_a[0]}) scores {pick_a[4]:.2f} on half B; '
          f'picked on half B ({pick_b[0]}) scores {pick_b[3]:.2f} on half A; mean {(pick_a[4] + pick_b[3]) / 2:.2f}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chosen = args.epochs if args.all_data else best[0]
    state, probs_out = states[chosen - 1], probs_by_epoch[chosen - 1]
    if args.swa_from:
        head.load_state_dict(average_states(states, args.swa_from))
        recompute_batchnorm(head, (stem(images.to(device)).float() for images, _ in train_loader))
        probs = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            for images, _ in val_loader:
                probs.append(torch.softmax(head(stem(images.to(device))).float(), 1).cpu())
        probs_out = torch.cat(probs).numpy(); pred = probs_out.argmax(1).tolist()
        print(f'SWA epochs {args.swa_from}-{args.epochs}: holdout macro-F1 {macro_f1(truth, pred):.2f} (clean '
              f'{macro_f1([t for t, i in zip(truth, val_ids) if i in clean], [p for p, i in zip(pred, val_ids) if i in clean]):.2f}; '
              f'halves {macro_f1([t for t, a in zip(truth, in_a) if a], [p for p, a in zip(pred, in_a) if a]):.2f} / '
              f'{macro_f1([t for t, a in zip(truth, in_a) if not a], [p for p, a in zip(pred, in_a) if not a]):.2f}) '
              f'against the last epoch {history[-1][1]:.2f}', flush=True)
        state, chosen = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, f'swa{args.swa_from}-{args.epochs}'
    np.save(args.output.with_suffix('.holdout_probs.npy'), probs_out)
    np.save(args.output.with_suffix('.holdout_probs_all_epochs.npy'), np.stack(probs_by_epoch))
    torch.save({'format': 'sar-branch-head-v1', 'split': args.split, 'init_from': str(args.init_from), 'epoch': chosen, 'all_data': args.all_data, 'views': args.views,
                'detector': str(args.detector), 'state_dict': state}, args.output)
    print(f'saved the epoch-{chosen} branch to {args.output}', flush=True)
    args.output.with_suffix('.json').write_text(json.dumps({'history': history, 'val_ids': val_ids}))


if __name__ == '__main__':
    main()
