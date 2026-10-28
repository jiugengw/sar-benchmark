"""Append the detector's own sub-0.001 boxes to a finished (rounded + hedged) submission, ranked last.

The submission writer keeps boxes down to score 0.001. The same model, predicted with conf 1e-5
(make_submission.py --conf 1e-5 --max-det 1000), gives the same boxes above 0.001 plus ~40 per image below it. Those
are appended below every box already in the ZIP, hedge copies included, keeping their order, so they can only add
recall: AP takes the best precision at each recall level, and boxes ranked last never lower it. Each image and class
still holds at most 100 boxes (the scorer's limit); the low boxes only fill what is left. Rounded like the originals
(round_boxes.py, whole-pixel-classes); classification.csv is copied byte for byte.

Holdout (tools/analysis/lowconf_tail_study.py, refit 340514, 28 Oct): official-mix 58.46 -> 58.55 (+0.09), new scenes
+0.09; placing them between the hedge tiers gave +0.11, a choice tuned on the holdout, so last it is.

    python models/yolo/lowconf_tail.py hedged.zip lowconf.zip out.zip
"""
from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))

from round_boxes import round_predictions  # noqa: E402


def key(p):
    return p['image_id'], p['category_id'], tuple(round(v, 3) for v in p['bbox']), round(p['score'], 6)


def main(hedged: Path, lowconf: Path, target: Path, images: Path = ROOT / 'data/course-data/val/images.json') -> None:
    with zipfile.ZipFile(hedged) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    boxes = json.loads(members['detection.json'])
    widths = {im['id']: im['width'] for im in json.loads(images.read_text())['images']}
    with zipfile.ZipFile(lowconf) as archive:
        raw = round_predictions(json.loads(archive.read('detection.json')), widths, 'whole-pixel-classes')
    # the low-conf run must be the ZIP's own model: its boxes >= 0.001 are exactly the ZIP's originals
    mine, theirs = Counter(map(key, (p for p in boxes if p['score'] >= 0.001))), Counter(map(key, (p for p in raw if p['score'] >= 0.001)))
    if mine != theirs:
        raise ValueError(f'boxes >= 0.001 differ: {sum((mine - theirs).values())} only in the ZIP, '
                         f'{sum((theirs - mine).values())} only in the low-conf run; not the same model or settings')
    floor = min(p['score'] for p in boxes)
    room = defaultdict(lambda: 100)
    for p in boxes:
        room[p['image_id'], p['category_id']] -= 1
    low = sorted((p for p in raw if p['score'] < 0.001), key=lambda p: -p['score'])
    scale = floor / 0.001 * 0.999  # every low box (score < 0.001) lands strictly below the lowest box in the ZIP
    added = []
    for p in low:
        k = p['image_id'], p['category_id']
        if room[k] > 0:
            room[k] -= 1
            added.append({**p, 'score': p['score'] * scale})
    members['detection.json'] = json.dumps(boxes + added).encode()
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    print(f'{target}: {len(boxes)} boxes + {len(added)} low boxes (of {len(low)}; lowest ZIP score {floor:.2e}, '
          f'added {min(p["score"] for p in added):.2e}-{max(p["score"] for p in added):.2e}); '
          'classification.csv byte-for-byte unchanged')


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:4]))
