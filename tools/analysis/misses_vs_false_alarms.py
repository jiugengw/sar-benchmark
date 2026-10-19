"""Split each class's AP50 loss into objects never found (100 - max recall) and false alarms ranked among the
real boxes (max recall - AP50). Clean holdout, memorise-lr2 model, NMS 0.7.

    python tools/analysis/misses_vs_false_alarms.py
"""
import json, sys, io, contextlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models/yolo")); sys.path.insert(0, str(ROOT))
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from postprocess_study import apply
from sarbench.twins import clean_image_ids
raw = str(ROOT / "results/pp-memorise-lr2/raw")
gt = json.load(open(raw + "/instances_val_holdout.json")); clean = sorted(clean_image_ids(gt["images"]))
preds = apply(json.load(open(raw + "/predictions_val.json")), "nms", 0.7)
with contextlib.redirect_stdout(io.StringIO()):
    g = COCO(raw + "/instances_val_holdout.json"); ev = COCOeval(g, g.loadRes([dict(p) for p in preds]), "bbox")
    ev.params.imgIds = clean; ev.evaluate(); ev.accumulate()
P = ev.eval["precision"][0, :, :, 0, -1]; R = ev.eval["recall"][0, :, 0, -1]
names = {c["id"]: c["name"] for c in gt["categories"]}
print(f"{'class':13s} {'AP50':>5s} {'max recall':>10s} {'lost to misses':>14s} {'lost to false alarms':>20s}")
tm = tf = 0
for k, c in enumerate(ev.params.catIds):
    ap50 = 100 * P[:, k].mean(); r = 100 * R[k]; tm += 100 - r; tf += r - ap50
    print(f"{names[c]:13s} {ap50:5.1f} {r:10.1f} {100 - r:14.1f} {r - ap50:20.1f}")
print(f"mAP points: misses {tm / 9:.1f}, false alarms {tf / 9:.1f}")
