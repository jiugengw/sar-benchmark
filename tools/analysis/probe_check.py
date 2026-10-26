"""Does a one-class submission score exactly that class's AP / 9 with the local copy of the official metric?

Run from the repository root (needs data/course-data and the holdout results on the cluster):

    python tools/analysis/probe_check.py results/ft40-s1r-memorise-337666/one2many
"""
import json, sys
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, 'models/yolo')
from sarbench.metrics import detection_metrics
from hedge_study import CHOSEN_TIERS, hedge_tiers
from round_boxes import round_predictions
r = Path(sys.argv[1]); gt_path = r / 'instances_val_holdout.json'
gt = json.loads(gt_path.read_text()); ids = [im['id'] for im in gt['images']]
cid = {c['name']: c['id'] for c in gt['categories']}
preds = hedge_tiers(round_predictions(json.loads((r / 'predictions_val.json').read_text()),
                                      {im['id']: im['width'] for im in gt['images']}, 'whole-pixel-classes'), CHOSEN_TIERS)
full = detection_metrics(str(gt_path), preds, ids)
print(f"all classes: mAP {100*full['mAP']:.3f}")
for name in ('bridge', 'playground'):
    one = detection_metrics(str(gt_path), [p for p in preds if p['category_id'] == cid[name]], ids)
    print(f"{name}-only: mAP {100*one['mAP']:.3f} x 9 = {900*one['mAP']:.2f}; {name} AP in the full run {100*full['per_class_AP'][name]:.2f}; "
          f"AP50 x 9 = {900*one['AP50']:.2f}")
