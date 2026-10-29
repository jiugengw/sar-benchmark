#!/bin/bash
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
R=results/ft40-s1r-memorise-337666/one2many; O=/tmp/flipcheck_jw
python /tmp/flip_value_jw.py $R/instances_val_holdout.json
python models/yolo/raw_candidates.py --weights runs/ft40-s1r-memorise-337666/train/weights/best.pt --images $O/turned.json \
  --split-root $O --out $O/pred_turned.json --iou 0.7 --max-det 300 --device cpu 2>&1 | grep RAW_CAND
python - <<'PY'
import json, sys
sys.path.insert(0, '.')
from sarbench.metrics import detection_metrics
O = '/tmp/flipcheck_jw'; R = 'results/ft40-s1r-memorise-337666/one2many'
asis = json.load(open(f'{O}/asis.json')); ids = [im['id'] for im in asis['images']]
pa = [p for p in json.load(open(f'{R}/predictions_val.json')) if p['image_id'] in set(ids)]
a = detection_metrics(f'{O}/asis.json', pa, ids)
t = detection_metrics(f'{O}/turned.json', json.load(open(f'{O}/pred_turned.json')), ids)
print(f"FLIPVALUE as they are: mAP {100*a['mAP']:.1f} | turned back to the training orientation: mAP {100*t['mAP']:.1f}")
for k in a['per_class_AP']:
    if a['per_class_AP'][k] >= 0: print(f"  {k:13s} {100*a['per_class_AP'][k]:5.1f} -> {100*t['per_class_AP'][k]:5.1f}")
PY
echo FLIPVALUE_DONE
