"""COCO annotation files for Dome-DETR on our 8453 / 939 holdout split (the split every YOLO run uses).

Boxes stay in original image pixels and point at the original JPEGs; Dome resizes each image to 512 x 512 itself.
Category ids become Dome label slots (see CATEGORY_TO_SLOT in common.py). The holdout ground truth is also written
with our own category ids, as instances_val_holdout.json, for sarbench.twins.score.

    python models/dome/prepare_data.py --out runs/dome-test/data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.dome.common import CATEGORY_TO_SLOT, DATA, SPLIT  # noqa: E402


def prepare(out: Path, data: Path = DATA, split_path: Path = SPLIT) -> dict:
    coco = json.loads((data / 'train/instances.json').read_text())
    split = json.loads(split_path.read_text())
    ids = {'train': set(map(int, split['train_image_ids'])), 'holdout': set(map(int, split['val_image_ids']))}
    if ids['train'] & ids['holdout'] or len(ids['train']) != 8453 or len(ids['holdout']) != 939:
        raise ValueError('expected the saved 8453 / 939 split with no overlap')
    names = {int(c['id']): c['name'] for c in coco['categories']}
    if sorted(names) != sorted(CATEGORY_TO_SLOT):
        raise ValueError(f'unexpected categories {names}')
    slots = [{'id': CATEGORY_TO_SLOT[i], 'name': names[i]} for i in sorted(names, key=CATEGORY_TO_SLOT.get)]
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, keep in ids.items():
        images = [im for im in coco['images'] if int(im['id']) in keep]
        annotations = [dict(id=int(a['id']), image_id=int(a['image_id']), category_id=CATEGORY_TO_SLOT[int(a['category_id'])],
                            bbox=[float(v) for v in a['bbox']], area=float(a['area']), iscrowd=int(a.get('iscrowd', 0)))
                       for a in coco['annotations'] if int(a['image_id']) in keep]
        (out / f'{name}.json').write_text(json.dumps({'images': images, 'annotations': annotations,
                                                       'categories': slots}) + '\n')
        counts[name] = {'images': len(images), 'boxes': len(annotations)}
    holdout = {'images': [im for im in coco['images'] if int(im['id']) in ids['holdout']],
               'annotations': [a for a in coco['annotations'] if int(a['image_id']) in ids['holdout']],
               'categories': coco['categories']}
    (out / 'instances_val_holdout.json').write_text(json.dumps(holdout) + '\n')
    print(f'Dome data in {out}: {counts}', flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.out)


if __name__ == '__main__':
    main()
