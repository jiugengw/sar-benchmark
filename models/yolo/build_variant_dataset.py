"""Build a variant of the 512 px YOLO dataset (data/yolo11s-sar-512) for one experiment.

Context boxes (labels change, images are hard-linked, not copied):
    python models/yolo/build_variant_dataset.py --out data/yolo-sar-512-ctx3 --context-classes bridge --context-factor 3
Every box of the chosen classes is enlarged around its centre by the factor (less if that would cross the image
border, so the centre never moves), so the detector learns the surroundings: river and road around a bridge. At
prediction time context_boxes.py shrinks those classes' boxes back by the same factor.

Filtered inputs (images change, labels are linked):
    python models/yolo/build_variant_dataset.py --out data/yolo-sar-512-water --filter water
Every image goes through input_filters.apply, the same function the submission code uses.

A random share of the training images (learning curve; labels and holdout unchanged):
    python models/yolo/build_variant_dataset.py --out data/yolo-sar-512-half --train-fraction 0.5
The share is drawn within each image class (seeded, the same images every time), so every class keeps its proportion.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.yolo.input_filters import FILTERS, apply  # noqa: E402

SOURCE = ROOT / 'data/yolo11s-sar-512'


def kept_train_files(image_map: dict, fraction: float, complement: bool = False) -> set[str]:
    """File names of the training share used by --train-fraction (seeded per image class), or of its complement."""
    by_class: dict[int, list[str]] = {}
    for path, info in sorted(image_map.items()):
        if path.startswith('images/train/'):
            by_class.setdefault(info['csv_image_label'], []).append(Path(path).name)
    kept = set()
    for label, files in sorted(by_class.items()):
        random.Random(label).shuffle(files)
        cut = round(len(files) * fraction)
        kept.update(files[cut:] if complement else files[:cut])
    return kept


def enlarge_line(line: str, classes: set[int], factor: float) -> str:
    """One YOLO label line (class cx cy w h, normalised); boxes of `classes` grow around their centre."""
    parts = line.split()
    if not parts or int(parts[0]) not in classes:
        return line
    cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
    w = min(w * factor, 2 * min(cx, 1 - cx))
    h = min(h * factor, 2 * min(cy, 1 - cy))
    return f'{cls} {cx:.9f} {cy:.9f} {w:.9f} {h:.9f}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--context-classes', default='', help='comma-separated class names to enlarge, e.g. bridge')
    parser.add_argument('--context-factor', type=float, default=3.0)
    parser.add_argument('--filter', choices=FILTERS, default='none')
    parser.add_argument('--train-fraction', type=float, default=1.0,
                        help='keep this share of the training images, drawn per image class')
    parser.add_argument('--complement', action='store_true',
                        help='keep the OTHER training images instead (e.g. the half not used by --train-fraction 0.5)')
    args = parser.parse_args()
    manifest = json.loads((args.source / 'conversion_manifest.json').read_text())
    names = manifest['classes']
    context = [n.strip() for n in args.context_classes.split(',') if n.strip()]
    if set(context) - set(names):
        raise ValueError(f'unknown classes {set(context) - set(names)}')
    if not context and args.filter == 'none' and args.train_fraction >= 1 and not args.complement:
        raise ValueError('nothing to change: give --context-classes, --filter and/or --train-fraction')
    if args.out.exists():
        raise FileExistsError(f'{args.out} exists; delete it first or choose another --out')
    out = args.out.resolve()
    class_ids = {names.index(n) for n in context}
    image_map = json.loads((args.source / 'image_map.json').read_text())
    kept = kept_train_files(image_map, args.train_fraction, args.complement)

    for split in ('train', 'val'):
        source_images, source_labels = args.source / 'images' / split, args.source / 'labels' / split
        (out / 'images').mkdir(parents=True, exist_ok=True)
        (out / 'labels').mkdir(parents=True, exist_ok=True)
        if class_ids:
            target = out / 'labels' / split
            target.mkdir()
            changed = 0
            for path in sorted(source_labels.glob('*.txt')):
                lines = path.read_text().splitlines()
                new = [enlarge_line(line, class_ids, args.context_factor) for line in lines]
                changed += sum(a != b for a, b in zip(lines, new))
                (target / path.name).write_text('\n'.join(new) + ('\n' if new else ''))
            print(f'{split}: enlarged {changed} boxes of {context} by up to {args.context_factor}x', flush=True)
        else:
            (out / 'labels' / split).symlink_to(source_labels.resolve(), target_is_directory=True)
        if args.filter != 'none':
            import numpy as np
            from PIL import Image

            target = out / 'images' / split
            target.mkdir()
            paths = [p for p in sorted(source_images.glob('*.png')) if split != 'train' or p.name in kept]
            for index, path in enumerate(paths, 1):
                with Image.open(path) as image:
                    gray = np.asarray(image.convert('L'), dtype=np.uint8)
                Image.fromarray(apply(args.filter, gray), 'RGB').save(target / path.name)
                if index % 2000 == 0:
                    print(f'{split}: filtered {index}/{len(paths)} images', flush=True)
            print(f'{split}: {len(paths)} images through the {args.filter!r} filter', flush=True)
        else:
            # A real folder of hard links, not a folder symlink: Ultralytics resolves the dataset's image folder, and
            # through a folder symlink it would find the ORIGINAL labels next to the original images.
            target = out / 'images' / split
            target.mkdir()
            for path in sorted(source_images.iterdir()):
                if split == 'train' and path.name not in kept:
                    continue
                try:
                    os.link(path, target / path.name)
                except OSError:  # another filesystem: fall back to a per-file link, which is not resolved
                    (target / path.name).symlink_to(path.resolve())

    for name in ('conversion_manifest.json', 'image_map.json'):
        shutil.copy2(args.source / name, out / name)
    names_yaml = ''.join(f'\n  {i}: {n}' for i, n in enumerate(names))
    (out / 'dataset.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\nnc: {len(names)}\nnames:{names_yaml}\n')
    (out / 'variant.json').write_text(json.dumps({
        'source': str(args.source.resolve()), 'context_classes': context, 'context_factor': args.context_factor,
        'filter': args.filter, 'train_fraction': args.train_fraction, 'complement': args.complement,
        'train_images': len(kept)}, indent=2) + '\n')
    print(f'VARIANT_READY {out / "dataset.yaml"}', flush=True)


if __name__ == '__main__':
    main()
