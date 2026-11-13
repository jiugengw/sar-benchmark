"""Score a submission ZIP locally against the val / test labels the TA released on 13 Nov 2026 (after the leaderboard
closed). Labels: data/course-data/released-labels/{val,test}_{labels.csv,instances.json} from
huggingface.co/datasets/doem1997/cs701-sar-course-data.

    ~/.venvs/sar-yolo/bin/python scripts/score_released.py submissions/phase2/phase2.zip test
    ~/.venvs/sar-yolo/bin/python scripts/score_released.py submissions/final/final.zip val --per-class
    ~/.venvs/sar-yolo/bin/python scripts/score_released.py ZIP test --images-file ids.txt   # a subset of image ids

Prints macro-F1, accuracy, mAP (COCO 0.5:0.95), AP50 and Δm. Δm uses the Phase 1 reference (F1 93.31, mAP 26.13)
for val and the Phase 2 reference (F1 93.99, mAP 25.41) for test, as Codabench did.
"""
import argparse, contextlib, csv, io, json, zipfile
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / 'data' / 'course-data' / 'released-labels'
REF = {'val': (0.9331, 0.2613), 'test': (0.9399, 0.2541)}
CLASSES = json.loads((LABELS / 'classes.json').read_text()) if (LABELS / 'classes.json').exists() else None


def class_names():
    """Names in label order (classes.json is a list of {category_id, label, name})."""
    return [c['name'] for c in sorted(CLASSES, key=lambda c: c['label'])]


def load_zip(path):
    with zipfile.ZipFile(path) as z:
        cls = {int(r['image_id']): int(r['label']) for r in csv.DictReader(io.TextIOWrapper(z.open('classification.csv')))}
        det = json.loads(z.read('detection.json'))
    return cls, det


def macro_f1(y, p, k=9):
    f = []
    for c in range(k):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c); fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        if tp + fp + fn: f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)), f


def score(cls, det, split, ids=None, per_class=False):
    gt_cls = {int(r['image_id']): int(r['label']) for r in csv.DictReader((LABELS / f'{split}_labels.csv').open())}
    ids = sorted(ids if ids is not None else gt_cls)
    y = [gt_cls[i] for i in ids]; p = [cls.get(i, -1) for i in ids]
    f1, f1s = macro_f1(y, p)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(LABELS / f'{split}_instances.json'))
        idset = set(ids); dt = gt.loadRes([d for d in det if d['image_id'] in idset] or [{'image_id': ids[0], 'category_id': 1, 'bbox': [0, 0, 1, 1], 'score': 0}])
        e = COCOeval(gt, dt, 'bbox'); e.params.imgIds = ids; e.evaluate(); e.accumulate(); e.summarize()
    mAP, ap50 = e.stats[0], e.stats[1]
    rf, rm = REF[split]
    out = {'images': len(ids), 'delta_m': round(50 * ((f1 / rf - 1) + (mAP / rm - 1)), 2), 'macro_f1': round(100 * f1, 2),
           'mAP': round(100 * mAP, 2), 'accuracy': round(100 * float(np.mean([a == b for a, b in zip(y, p)])), 2),
           'AP50': round(100 * ap50, 2)}
    if per_class:
        names = class_names(); prec = e.eval['precision']  # [T, R, K, A, M]
        out['per_class'] = {}
        for k, cid in enumerate(e.params.catIds):
            pk = prec[:, :, k, 0, -1]; pk = pk[pk > -1]
            pk50 = prec[0, :, k, 0, -1]; pk50 = pk50[pk50 > -1]
            out['per_class'][names[cid - 1] if names else cid] = {
                'AP': round(100 * float(pk.mean()), 2) if pk.size else None,
                'AP50': round(100 * float(pk50.mean()), 2) if pk50.size else None,
                'F1': round(100 * f1s[cid - 1], 2) if cid - 1 < len(f1s) else None,
                'images': sum(1 for a in y if a == cid - 1)}
        conf = np.zeros((9, 10), int)
        for a, b in zip(y, p): conf[a, b if b >= 0 else 9] += 1
        out['confusion_rows_true_cols_pred'] = conf.tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip'); ap.add_argument('split', choices=['val', 'test'])
    ap.add_argument('--per-class', action='store_true'); ap.add_argument('--images-file')
    a = ap.parse_args()
    ids = [int(x) for x in Path(a.images_file).read_text().split()] if a.images_file else None
    cls, det = load_zip(a.zip)
    print(json.dumps(score(cls, det, a.split, ids, a.per_class), indent=1))


if __name__ == '__main__':
    main()
