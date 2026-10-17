"""Near-duplicate twin images in the labeled course training set, and the clean holdout subset.

Some scenes occur twice among the labeled training images: same framing and image class, usually identical boxes,
different speckle. A holdout image whose twin is a training image is nearly a training image, so holdout scores are
also reported on the clean subset: the holdout images with no twin among the training images.

    python -m sarbench.twins build --data data/course-data       # rewrite sarbench/train_twins.json
    python -m sarbench.twins query --data data/course-data --images data/course-data/val/images
    python -m sarbench.twins score results/<run>/seed_0          # full and clean scores of a saved run
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

TWINS_PATH = Path(__file__).with_name('train_twins.json')
THUMBNAIL = 64  # side of the compared grayscale thumbnails, in pixels
THRESHOLD = 0.9  # Pearson correlation of two thumbnails above which the images are twins


def thumbnails(paths):
    """Zero-mean, unit-norm grayscale thumbnails, one row per image, so a row product is a Pearson correlation."""
    from PIL import Image

    rows = np.stack([np.asarray(Image.open(path).convert('L').resize((THUMBNAIL, THUMBNAIL), Image.BILINEAR),
                                dtype=np.float32).ravel() for path in paths])
    rows -= rows.mean(axis=1, keepdims=True)
    return rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-6)


def find_twins(rows, labels, threshold=THRESHOLD, chunk=1024):
    """Pairs (i, j, correlation), i < j, of same-class thumbnail rows correlated above `threshold`."""
    pairs = []
    for start in range(0, len(rows), chunk):
        block = rows[start:start + chunk] @ rows.T
        for i, j in zip(*np.nonzero(block > threshold)):
            if start + i < j and labels[start + i] == labels[j]:
                pairs.append((int(start + i), int(j), float(block[i, j])))
    return pairs


def load_twins(path=TWINS_PATH):
    """File name -> set of twin file names, both as in train/instances.json (e.g. 'images/00079.jpg')."""
    twins = {}
    for pair in json.loads(Path(path).read_text())['pairs']:
        a, b = pair['images']
        twins.setdefault(a, set()).add(b)
        twins.setdefault(b, set()).add(a)
    return twins


def clean_image_ids(images, twins=None):
    """Ids of the COCO `images` with no twin among the training images. Every labeled image outside `images` counts
    as a training image, as in the 90/10 holdout split and the cross-validation folds."""
    twins = load_twins() if twins is None else twins
    evaluated = {image['file_name'] for image in images}
    return [image['id'] for image in images if twins.get(image['file_name'], set()) <= evaluated]


def build(data):
    """Find every same-class twin pair among the labeled training images and write them to TWINS_PATH."""
    coco = json.loads((data / 'train' / 'instances.json').read_text())
    label = {annotation['image_id']: annotation['category_id'] for annotation in coco['annotations']}
    images = sorted(coco['images'], key=lambda image: image['file_name'])
    rows = thumbnails([data / 'train' / image['file_name'] for image in images])
    pairs = find_twins(rows, [label[image['id']] for image in images])
    TWINS_PATH.write_text(json.dumps({
        'description': f'Same-class labeled training images whose {THUMBNAIL} x {THUMBNAIL} grayscale thumbnails have '
                       f'Pearson correlation above {THRESHOLD}. Written by python -m sarbench.twins build.',
        'images': len(images),
        'pairs': [{'images': [images[i]['file_name'], images[j]['file_name']], 'correlation': round(c, 4)}
                  for i, j, c in pairs],
    }, indent=1) + '\n')
    print(f'{len(pairs)} twin pairs among {len(images)} images, '
          f'{len({images[k]["file_name"] for i, j, _ in pairs for k in (i, j)})} images with a twin: {TWINS_PATH}')


def query(data, folder, output=None):
    """Report the images in `folder` (e.g. the unlabeled course validation set) with a twin among the training
    images. Without labels, the class is not compared."""
    coco = json.loads((data / 'train' / 'instances.json').read_text())
    train = sorted(image['file_name'] for image in coco['images'])
    queried = sorted(path for path in folder.rglob('*') if path.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    reference = thumbnails([data / 'train' / name for name in train])
    matches = []
    for start in range(0, len(queried), 1024):
        block = thumbnails(queried[start:start + 1024]) @ reference.T
        for path, j, c in zip(queried[start:start + 1024], block.argmax(axis=1), block.max(axis=1)):
            if c > THRESHOLD:
                matches.append({'image': str(path.relative_to(folder)), 'train_twin': train[j],
                                'correlation': round(float(c), 4)})
    print(f'{len(matches)} of {len(queried)} images in {folder} have a training twin')
    if output:
        Path(output).write_text(json.dumps(matches, indent=1) + '\n')


def score(results, data):
    """Full and clean holdout scores of a saved run: COCO detection metrics, plus macro-F1 and delta m when per-image
    classifications were saved (image_classification_predictions.json)."""
    from sarbench.metrics import classification_metrics, detection_metrics

    gt_path = results / 'instances_val_holdout.json'
    images = json.loads(gt_path.read_text())['images']
    predictions = json.loads((results / 'predictions_val.json').read_text())
    classes_path = results / 'image_classification_predictions.json'
    classified = {}
    if classes_path.is_file():  # {image id: class index} or {image id: {"label": class index, ...}}
        classified = {int(k): int(v['label'] if isinstance(v, dict) else v)
                      for k, v in json.loads(classes_path.read_text()).items()}
    with (data / 'train' / 'labels.csv').open(newline='') as stream:
        truth = {int(row['image_id']): int(row['label']) for row in csv.DictReader(stream)}
    report = {}
    for name, ids in (('full', [image['id'] for image in images]), ('clean', clean_image_ids(images))):
        scores = {'images': len(ids), **detection_metrics(gt_path, predictions, ids)}
        if classified:
            scores.update(classification_metrics([truth[i] for i in ids], [classified[i] for i in ids]))
            scores['delta_m'] = 50 * (scores['macro_f1'] / 0.9331 + scores['mAP'] / 0.2613 - 2)
        report[name] = scores
        print(f'{name:5s} {len(ids):4d} images: mAP {scores["mAP"]:.4f} AP50 {scores["AP50"]:.4f}'
              + (f' macro_f1 {scores["macro_f1"]:.4f} Δm {scores["delta_m"]:+.2f}' if classified else ''))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('build', 'query', 'score'))
    parser.add_argument('results', nargs='?', type=Path, help='score: a run results folder')
    parser.add_argument('--data', type=Path, default=Path('data/course-data'))
    parser.add_argument('--images', type=Path, help='query: folder of images to compare with the training set')
    parser.add_argument('--out', type=Path, help='query: write the matches to this JSON file')
    args = parser.parse_args()
    if args.command == 'build':
        build(args.data)
    elif args.command == 'query':
        if args.images is None:
            parser.error('query needs --images')
        query(args.data, args.images, args.out)
    else:
        if args.results is None:
            parser.error('score needs a results folder')
        score(args.results, args.data)


if __name__ == '__main__':
    main()
