#!/bin/bash
# Training-set fit: does the training-split stride-1 memorise model (337666) find bridges / playgrounds on images it
# was trained on? 250 training (non-holdout) images that contain them, CPU.
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
O=/tmp/trainfit_jw; mkdir -p $O
python - <<'PY'
import json, random
hold = json.load(open('results/ft40-s1r-memorise-337666/one2many/instances_val_holdout.json'))
tr = json.load(open('data/course-data/train/instances.json'))
hf = {im['file_name'] for im in hold['images']}
ids = sorted({a['image_id'] for a in tr['annotations'] if a['category_id'] in (3, 7)})
ims = [im for im in tr['images'] if im['id'] in set(ids) and im['file_name'] not in hf]
random.Random(0).shuffle(ims); ims = ims[:250]; keep = {im['id'] for im in ims}
json.dump({'images': ims, 'annotations': [a for a in tr['annotations'] if a['image_id'] in keep], 'categories': tr['categories']},
          open('/tmp/trainfit_jw/gt.json', 'w'))
print(len(ims), 'training images')
PY
python models/yolo/raw_candidates.py --weights runs/ft40-s1r-memorise-337666/train/weights/best.pt --images $O/gt.json \
  --split-root data/course-data/train --out $O/pred.json --iou 0.7 --max-det 300 --device cpu 2>&1 | grep -v "Warning\|_C._set" | tail -2
python - <<'PY'
import json, sys
sys.path.insert(0, '.')
from sarbench.metrics import detection_metrics
gt = json.load(open('/tmp/trainfit_jw/gt.json')); ids = [im['id'] for im in gt['images']]
m = detection_metrics('/tmp/trainfit_jw/gt.json', json.load(open('/tmp/trainfit_jw/pred.json')), ids)
print('TRAINFIT on 250 training images it learned from:', {k: round(100 * v, 1) for k, v in m['per_class_AP'].items() if v >= 0})
for c in (3, 7):
    one = detection_metrics('/tmp/trainfit_jw/gt.json', [p for p in json.load(open('/tmp/trainfit_jw/pred.json')) if p['category_id'] == c], ids)
    print('  class', c, 'AP50', round(900 * one['AP50'] / len([k for k, v in m['per_class_AP'].items() if v >= 0]) * len([k for k, v in m['per_class_AP'].items() if v >= 0]) / 9, 1))
PY
echo TRAINFIT_DONE
