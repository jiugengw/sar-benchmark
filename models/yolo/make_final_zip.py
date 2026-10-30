#!/usr/bin/env python3
"""Make the submission ZIP from the saved final model, training nothing.

The same five steps that produced submissions/alldata-s1fsr2-rounded-hedged-s1two-tail/ (Δm +65.92), in one command:
  1. detector boxes: make_submission.py, one2many branch, boxes >= 0.001, at most 300 per image
  2. rounding to whole pixels: round_boxes.py (whole-pixel-classes)
  3. hedging: resized and shifted copies of each box at lower scores (hedge_study.py)
  4. low-score tail: the detector's boxes below 0.001, ranked last (make_submission.py --conf 1e-5 --max-det 1000,
     then lowconf_tail.py, which checks that the boxes >= 0.001 are the ones of step 1)
  5. image labels: the saved two-head classifier (branch head + conv head, probabilities averaged 50/50, then the
     size rule of classifier_combo_study.py)
The ZIP holds detection.json (steps 1-4) and classification.csv (step 5).

Run from the repository root on a CUDA GPU, after the setup in docs/TEAM_GUIDE.md section 1, with the weights in
final_model/ (final_model/README.md says where to download them):
    python models/yolo/make_final_zip.py --out final.zip                         # Phase 1 images (data/course-data/val)
    python models/yolo/make_final_zip.py --images NEW_DIR --out phase2.zip       # NEW_DIR: images.json + the images
    python models/yolo/make_final_zip.py --out final.zip --compare submissions/alldata-s1fsr2-rounded-hedged-s1two-tail/alldata-s1fsr2-rounded-hedged-s1two-tail.zip
NEW_DIR is laid out like data/course-data/val: an images.json whose "images" list id, file_name, width and height,
and the image files at NEW_DIR/<file_name>. Intermediate ZIPs and probabilities are kept in <out>-work/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / 'models/yolo/make_submission.py').is_file():
    sys.exit('Run from the repository root.')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))
WEIGHTS = ROOT / 'final_model'


def run(*args) -> None:
    subprocess.run([sys.executable, '-u', *map(str, args)], cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_zip(path: Path) -> tuple[list, dict]:
    with zipfile.ZipFile(path) as archive:
        boxes = json.loads(archive.read('detection.json'))
        rows = archive.read('classification.csv').decode().strip().splitlines()[1:]
    return boxes, dict(row.split(',') for row in rows)


def pair_boxes(boxes: list, ref_boxes: list, tolerance: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Pair boxes of the same image and class whose edges are all within `tolerance` px, closest pairs first.
    Returns the largest edge difference and the score difference of each pair."""
    groups = defaultdict(lambda: ([], []))
    for side, items in ((0, boxes), (1, ref_boxes)):
        for p in items:
            groups[p['image_id'], p['category_id']][side].append(p)
    edges, scores = [], []
    for mine, ref in groups.values():
        if not mine or not ref:
            continue
        gap = np.abs(np.array([p['bbox'] for p in mine])[:, None] - np.array([p['bbox'] for p in ref])[None]).max(2)
        used_mine, used_ref = set(), set()
        for i, j in zip(*np.unravel_index(np.argsort(gap, axis=None), gap.shape)):
            if gap[i, j] > tolerance + 1e-6:
                break
            if i in used_mine or j in used_ref:
                continue
            used_mine.add(i)
            used_ref.add(j)
            edges.append(gap[i, j])
            scores.append(abs(mine[i]['score'] - ref[j]['score']))
    return np.array(edges), np.array(scores)


def compare(mine: Path, reference: Path) -> None:
    """Print how far the new ZIP is from a reference ZIP: MATCH (identical), CLOSE MATCH (same labels, boxes differ
    only by a different GPU's rounding) or DIFFERENT."""
    boxes, labels = read_zip(mine)
    ref_boxes, ref_labels = read_zip(reference)
    changed = sum(labels.get(i) != k for i, k in ref_labels.items())
    print(f'labels: {changed} of {len(ref_labels)} images differ from the reference')
    if boxes == ref_boxes:
        print(f'boxes: identical to the reference ({len(boxes)} boxes, same order and scores)')
        print('MATCH' if not changed else 'DIFFERENT', flush=True)
        return
    # Another GPU type (or driver) gives the same model slightly different floating-point results: box edges move by
    # about 0.001 px, whole-pixel rounding can then flip by 1 px, and boxes at the score cut-offs can come or go.
    edges, scores = pair_boxes(boxes, ref_boxes)
    unpaired = max(len(boxes), len(ref_boxes)) - len(edges)
    print(f'boxes: {len(boxes)} vs {len(ref_boxes)} in the reference; {len(edges)} pair up within 1 px '
          f'(median edge difference {np.median(edges):.1e} px, median score difference {np.median(scores):.1e}, '
          f'largest {scores.max():.1e}); {unpaired} without a partner')
    close = not changed and unpaired <= 0.01 * len(ref_boxes) and np.median(edges) < 0.01
    print('CLOSE MATCH: same labels; boxes differ only by GPU rounding' if close else 'DIFFERENT', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', type=Path, default=ROOT / 'data/course-data/val',
                        help='folder with images.json and the images (default: the Phase 1 test images)')
    parser.add_argument('--out', type=Path, required=True, help='ZIP to write; existing files are not overwritten')
    parser.add_argument('--detector', type=Path, default=WEIGHTS / 'detector.pt')
    parser.add_argument('--branch', type=Path, default=WEIGHTS / 'branch_head.pt')
    parser.add_argument('--conv', type=Path, default=WEIGHTS / 'conv_head.pt')
    parser.add_argument('--compare', type=Path, help='reference ZIP to compare the result with')
    args = parser.parse_args()
    from classifier_combo_study import apply_size_rule, size_allowed
    from hedge_study import rewrite_zip as hedge_zip
    from lowconf_tail import main as add_tail
    from round_boxes import rewrite_zip as round_zip

    images_dir = args.images.resolve()
    images_json = images_dir / 'images.json'
    images = json.loads(images_json.read_text())['images']
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    work = args.out.with_name(args.out.stem + '-work')
    work.mkdir(parents=True, exist_ok=True)
    for name in ('detector', 'branch', 'conv'):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f'{path} (download the weights: final_model/README.md)')
        print(f'{name}: {path} sha256 {sha256(path)}', flush=True)
    print(f'{len(images)} images from {images_dir}', flush=True)
    # The model was trained and packaged without an input filter; make sure none leaks in from the environment.
    os.environ['SAR_INPUT_FILTER'] = 'none'
    os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'models/yolo'), os.environ.get('PYTHONPATH', '')])

    # make_submission.py reads <data>/val (the images) and <data>/train/labels.csv (a fallback label it needs to start)
    data = work / 'data'
    data.mkdir(exist_ok=True)
    for name, target in (('val', images_dir), ('train', ROOT / 'data/course-data/train')):
        link = data / name
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target)

    base, rounded, hedged = work / 'base.zip', work / 'rounded.zip', work / 'hedged.zip'
    lowconf, tailed = work / 'lowconf.zip', work / 'tailed.zip'
    for path in (base, rounded, hedged, lowconf, tailed):
        path.unlink(missing_ok=True)
    print('== 1. detector boxes', flush=True)
    run(ROOT / 'models/yolo/make_submission.py', '--weights', args.detector, '--data', data,
        '--detection-branch', 'one2many', '--out', base)
    print('== 2. rounding', flush=True)
    round_zip(base, rounded, 'whole-pixel-classes', images_json)
    print('== 3. hedging', flush=True)
    hedge_zip(rounded, hedged)
    print('== 4. low-score tail', flush=True)
    run(ROOT / 'models/yolo/make_submission.py', '--weights', args.detector, '--data', data,
        '--detection-branch', 'one2many', '--out', lowconf, '--conf', '1e-5', '--max-det', '1000')
    add_tail(hedged, lowconf, tailed, images_json)
    print('== 5. two-head classifier', flush=True)
    probs = {}
    for name, script, weights in (('branch', 'branch_head_predict.py', args.branch),
                                  ('conv', 'conv_head_predict.py', args.conv)):
        probs[name] = work / f'probs_{name}.json'
        run(ROOT / 'models/yolo' / script, '--detector', args.detector, '--branch', weights,
            '--images', images_json, '--split-root', images_dir, '--out', probs[name])
    b, c = (json.loads(probs[name].read_text()) for name in ('branch', 'conv'))
    ids = [im['id'] for im in images]
    mean = 0.5 * np.array([b[str(i)] for i in ids]) + 0.5 * np.array([c[str(i)] for i in ids])
    labels = apply_size_rule(mean, [(im['width'], im['height']) for im in images], size_allowed()).argmax(1)

    with zipfile.ZipFile(tailed) as archive:
        detection = archive.read('detection.json')
    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('detection.json', detection)
        archive.writestr('classification.csv', 'image_id,label\n' + ''.join(f'{i},{int(k)}\n' for i, k in zip(ids, labels)))
    print(f'FINAL_ZIP {args.out}: {len(json.loads(detection))} boxes, {len(ids)} image labels', flush=True)
    if args.compare:
        compare(args.out, args.compare)


if __name__ == '__main__':
    main()
