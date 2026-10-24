"""Combine the two image-classification heads of one model and the image-size rule; holdout macro-F1, cross-fitted.

Heads: the mean-pooled 1024-unit head (current submissions) and the convolutional branch (conv_head_study.py), both on
the same frozen detector. p = w * p_conv + (1 - w) * p_mean; then the size rule: classes never seen at the image's
original size in the training split get probability 0 (e.g. 256 px images hold only bridge, ship, harbour, oil tank
or aircraft; 224 px only aircraft; every other size only ships). w is chosen on one holdout half (twin groups
together) and scored on the other.

    python models/yolo/classifier_combo_study.py --mean runs/<pkg>/holdout_probs.json --conv runs/<conv>/conv_head.holdout_probs.npy
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))


def macro_f1(truth, pred):
    f1s = []
    for c in set(truth) | set(pred):
        tp = sum(1 for t, p in zip(truth, pred) if t == c == p)
        fp = sum(1 for t, p in zip(truth, pred) if p == c != t)
        fn = sum(1 for t, p in zip(truth, pred) if t == c != p)
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return 100 * float(np.mean(f1s))


def size_allowed() -> dict:
    """Original image size -> set of 0-based classes seen at that size in the training split."""
    train = json.loads((ROOT / 'data/course-data/train/instances.json').read_text())
    split = json.loads((ROOT / 'runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json').read_text())
    held = set(map(int, split['val_image_ids']))
    size = {im['id']: (im['width'], im['height']) for im in train['images']}
    allowed = defaultdict(set)
    for a in train['annotations']:
        if a['image_id'] not in held:
            allowed[size[a['image_id']]].add(a['category_id'] - 1)
    return allowed


def apply_size_rule(probs: np.ndarray, sizes: list[tuple[int, int]], allowed: dict) -> np.ndarray:
    out = probs.copy()
    for k, s in enumerate(sizes):
        ok = allowed.get(s) if s in ((512, 512), (256, 256), (224, 224)) else {7}  # any other size: ship
        if ok:
            out[k] *= np.array([c in ok for c in range(out.shape[1])], dtype=float)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mean', type=Path, required=True, help='holdout_probs.json from head_probs.py')
    parser.add_argument('--conv', type=Path, help='holdout probabilities (.npy, holdout split order) of the conv branch')
    args = parser.parse_args()
    from ranking_study import twin_groups
    from sarbench.twins import clean_image_ids

    split = json.loads((ROOT / 'runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json').read_text())
    val_ids = list(map(int, split['val_image_ids']))
    with (ROOT / 'data/course-data/train/labels.csv').open(newline='') as stream:
        labels = {int(r['image_id']): int(r['label']) for r in csv.DictReader(stream)}
    train = json.loads((ROOT / 'data/course-data/train/instances.json').read_text())
    size = {im['id']: (im['width'], im['height']) for im in train['images']}
    held_images = [im for im in train['images'] if im['id'] in set(val_ids)]
    clean = set(clean_image_ids(held_images))
    groups = twin_groups(held_images)
    unique = sorted(set(groups.values()))
    half = set(np.random.default_rng(0).permutation(unique)[: len(unique) // 2])
    in_a = np.array([groups[i] in half for i in val_ids])
    is_clean = np.array([i in clean for i in val_ids])
    truth = [labels[i] for i in val_ids]
    sizes = [size[i] for i in val_ids]
    allowed = size_allowed()
    mean = json.loads(args.mean.read_text())
    p_mean = np.array([mean[str(i)] for i in val_ids])
    p_conv = np.load(args.conv) if args.conv else None

    def report(name, probs):
        pred = probs.argmax(1).tolist()
        sel = lambda m: ([t for t, k in zip(truth, m) if k], [p for p, k in zip(pred, m) if k])
        print(f'  {name:42s} full {macro_f1(truth, pred):.2f}  clean {macro_f1(*sel(is_clean)):.2f}  '
              f'halves {macro_f1(*sel(in_a)):.2f} / {macro_f1(*sel(~in_a)):.2f}', flush=True)
        return macro_f1(*sel(in_a)), macro_f1(*sel(~in_a))

    report('mean head', p_mean)
    report('mean head + size rule', apply_size_rule(p_mean, sizes, allowed))
    if p_conv is None:
        return
    report('conv branch', p_conv)
    report('conv branch + size rule', apply_size_rule(p_conv, sizes, allowed))
    rows = []
    for w in np.linspace(0, 1, 11):
        rows.append((w, report(f'w {w:.1f} conv + size rule', apply_size_rule(w * p_conv + (1 - w) * p_mean, sizes, allowed))))
    picked = []
    for pick, rep in ((0, 1), (1, 0)):
        best = max(rows, key=lambda r: r[1][pick])
        picked.append(best[1][rep])
        print(f'w chosen on half {"AB"[pick]}: {best[0]:.1f} -> half {"AB"[rep]}: {best[1][rep]:.2f}')
    print(f'cross-fitted macro-F1 of the combination: {np.mean(picked):.2f}')


if __name__ == '__main__':
    main()
