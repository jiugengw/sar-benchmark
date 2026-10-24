#!/usr/bin/env python3
"""Synthetic other-polarisation views of training images, learned from the training split only.

Many M4-SAR scenes were imaged in two polarisations (VH and VV), and the course split them independently, so a test
image is often the other polarisation of a scene the detector memorised. The detector half-recognises such scenes
(holdout mAP 55 vs 35 without a twin in training). This makes it see both views of every scene it learns.

1. Pairs: training images with IDENTICAL label sets (same classes and boxes to 0.01 px) show the same scene; only
   course labels are used (745 pairs).
2. Translator: a small U-Net maps one image of a pair to the other, both ways (it has to tell the direction from the
   image itself), on random 256 px crops with joint flips; L1 loss.
3. Synthesis: every 512 px training image of the M4-like classes gets a translated view plus fresh multiplicative
   speckle (the translator predicts the speckle-free mean), with the original labels. Images go into ONE zip file
   (cluster file quota) and a label list, read during training by train_with_extra.py.

    python models/yolo/pol_translate.py --out runs/pol-views
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
M4_LIKE = {2, 3, 5, 6, 7, 9}  # COCO ids: airport, bridge, harbor, oil_tank, playground, wind_turbine


def conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.down = nn.ModuleList([conv(1, c[0]), conv(c[0], c[1]), conv(c[1], c[2]), conv(c[2], c[3])])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(3)])
        self.merge = nn.ModuleList([conv(2 * c[i], c[i]) for i in range(3)])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x)
            if i < 3:
                skips.append(x)
                x = nn.functional.max_pool2d(x, 2)
        for i in reversed(range(3)):
            x = self.merge[i](torch.cat([self.up[i](x), skips[i]], 1))
        return torch.sigmoid(self.out(x))


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def same_scene_pairs(train_ids: set[int]) -> list[tuple[str, str]]:
    coco = json.loads((ROOT / 'data/course-data/train/instances.json').read_text())
    files = {im['id']: im['file_name'] for im in coco['images']}
    sizes = {im['id']: im['width'] for im in coco['images']}
    labels = defaultdict(list)
    for a in coco['annotations']:
        labels[a['image_id']].append((a['category_id'],) + tuple(round(v, 2) for v in a['bbox']))
    groups = defaultdict(list)
    for i, l in labels.items():
        if i in train_ids and sizes[i] == 512:
            groups[tuple(sorted(l))].append(i)
    return [(files[a], files[b]) for g in groups.values() for a in g for b in g if a != b]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--looks', type=float, default=4.0, help='speckle strength of the synthetic views')
    args = parser.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device('cuda:0')
    split = json.loads((ROOT / 'runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json').read_text())
    train_ids = set(map(int, split['train_image_ids']))
    pairs = same_scene_pairs(train_ids)
    data = ROOT / 'data/course-data/train'
    cache = {}
    for a, b in pairs:
        for f in (a, b):
            if f not in cache:
                cache[f] = load(data / f)
    random.shuffle(pairs)
    val_pairs, fit_pairs = pairs[: len(pairs) // 10], pairs[len(pairs) // 10:]
    print(f'{len(pairs)} ordered same-scene pairs ({len(fit_pairs)} fit, {len(val_pairs)} check)', flush=True)
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = args.epochs * (len(fit_pairs) // 16)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=steps)

    def batch(selection, crop=256):
        xs, ys = [], []
        for a, b in selection:
            x, y = cache[a], cache[b]
            i, j = np.random.randint(0, 512 - crop + 1, 2)
            x, y = x[i:i + crop, j:j + crop], y[i:i + crop, j:j + crop]
            if random.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            if random.random() < 0.5:
                x, y = x[::-1], y[::-1]
            xs.append(x.copy()); ys.append(y.copy())
        return (torch.from_numpy(np.stack(xs))[:, None].to(device), torch.from_numpy(np.stack(ys))[:, None].to(device))

    def check():
        model.eval()
        with torch.no_grad():
            x, y = batch(val_pairs[:64], 512)
            pred = model(x)
            return float((pred - y).abs().mean()), float((x - y).abs().mean())

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(fit_pairs)
        for k in range(0, len(fit_pairs) - 15, 16):
            x, y = batch(fit_pairs[k:k + 16])
            loss = (model(x) - y).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        l1, identity = check()
        print(f'epoch {epoch}: held-out pair L1 {l1:.4f} (copying the input: {identity:.4f})', flush=True)

    # synthesise views of every 512 px M4-like training image
    coco = json.loads((ROOT / 'data/course-data/train/instances.json').read_text())
    image_map = json.loads((ROOT / 'data/yolo11s-sar-512/image_map.json').read_text())
    yolo_name = {int(info['image_id']): Path(rel).name for rel, info in image_map.items()}
    by_image = defaultdict(list)
    for a in coco['annotations']:
        by_image[a['image_id']].append(a)
    targets = [im for im in coco['images'] if im['id'] in train_ids and im['width'] == 512
               and by_image[im['id']] and by_image[im['id']][0]['category_id'] in M4_LIKE]
    args.out.mkdir(parents=True, exist_ok=True)
    zpath = args.out / 'views.zip'
    labels_out = []
    model.eval()
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_STORED) as archive, torch.no_grad():
        for start in range(0, len(targets), 16):
            group = targets[start:start + 16]
            x = torch.from_numpy(np.stack([load(data / im['file_name']) for im in group]))[:, None].to(device)
            means = model(x)[:, 0].cpu().numpy()
            for im, mean in zip(group, means):
                noisy = mean * np.random.gamma(args.looks, 1.0 / args.looks, mean.shape)
                pixels = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
                buffer = io.BytesIO()
                Image.fromarray(pixels).save(buffer, format='PNG')
                name = f"views/{yolo_name.get(im['id'], Path(im['file_name']).stem + '.png')}"
                archive.writestr(name, buffer.getvalue())
                rows = []
                for a in by_image[im['id']]:
                    x0, y0, w, h = a['bbox']
                    rows.append([a['category_id'] - 1, (x0 + w / 2) / 512, (y0 + h / 2) / 512, w / 512, h / 512])
                r = np.array(rows, dtype=np.float32)
                labels_out.append({'im_file': f'zip://{name}', 'shape': (512, 512), 'cls': r[:, :1], 'bboxes': r[:, 1:],
                                   'segments': [], 'keypoints': None, 'normalized': True, 'bbox_format': 'xywh'})
    with open(args.out / 'views_labels.pkl', 'wb') as fh:
        pickle.dump(labels_out, fh)
    torch.save(model.state_dict(), args.out / 'translator.pt')
    print(f'POL_VIEWS {len(labels_out)} synthetic views -> {zpath}', flush=True)


if __name__ == '__main__':
    main()
