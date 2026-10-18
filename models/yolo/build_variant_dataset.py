"""Build a variant of the 512 px YOLO dataset (data/yolo11s-sar-512) for one experiment.

Context boxes (labels change, images are linked, not copied):
    python models/yolo/build_variant_dataset.py --out data/yolo-sar-512-ctx3 --context-classes bridge --context-factor 3
Every box of the chosen classes is enlarged around its centre by the factor (less if that would cross the image
border, so the centre never moves), so the detector learns the surroundings: river and road around a bridge. At
prediction time context_boxes.py shrinks those classes' boxes back by the same factor.

Filtered inputs (images change, labels are linked):
    python models/yolo/build_variant_dataset.py --out data/yolo-sar-512-water --filter water
Every image goes through input_filters.apply, the same function the submission code uses.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.yolo.input_filters import FILTERS, apply  # noqa: E402

SOURCE = ROOT / 'data/yolo11s-sar-512'


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
    args = parser.parse_args()
    manifest = json.loads((args.source / 'conversion_manifest.json').read_text())
    names = manifest['classes']
    context = [n.strip() for n in args.context_classes.split(',') if n.strip()]
    if set(context) - set(names):
        raise ValueError(f'unknown classes {set(context) - set(names)}')
    if not context and args.filter == 'none':
        raise ValueError('nothing to change: give --context-classes and/or --filter')
    if args.out.exists():
        raise FileExistsError(f'{args.out} exists; delete it first or choose another --out')
    out = args.out.resolve()
    class_ids = {names.index(n) for n in context}

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
            paths = sorted(source_images.glob('*.png'))
            for index, path in enumerate(paths, 1):
                with Image.open(path) as image:
                    gray = np.asarray(image.convert('L'), dtype=np.uint8)
                Image.fromarray(apply(args.filter, gray), 'RGB').save(target / path.name)
                if index % 2000 == 0:
                    print(f'{split}: filtered {index}/{len(paths)} images', flush=True)
            print(f'{split}: {len(paths)} images through the {args.filter!r} filter', flush=True)
        else:
            (out / 'images' / split).symlink_to(source_images.resolve(), target_is_directory=True)

    for name in ('conversion_manifest.json', 'image_map.json'):
        shutil.copy2(args.source / name, out / name)
    names_yaml = ''.join(f'\n  {i}: {n}' for i, n in enumerate(names))
    (out / 'dataset.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\nnc: {len(names)}\nnames:{names_yaml}\n')
    (out / 'variant.json').write_text(json.dumps({
        'source': str(args.source.resolve()), 'context_classes': context, 'context_factor': args.context_factor,
        'filter': args.filter}, indent=2) + '\n')
    print(f'VARIANT_READY {out / "dataset.yaml"}', flush=True)


if __name__ == '__main__':
    main()
