#!/bin/bash
# When the stride-1 branch classifier (job 337761) is saved: score it on the holdout next to the two-head, predict the
# official validation images with it (CPU), and build a single-model ZIP: M stride-1 boxes (rounded + hedged) with
# image labels from M stride-1's own branch classifier (size rule applied).
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
OUT=runs/sar-s1heads-337761.out
until grep -q "saved the epoch-.* branch" $OUT 2>/dev/null; do grep -q "Traceback" $OUT && { echo FAILED; exit 1; }; sleep 60; done
B=$(grep -o "runs/s1-heads-337761/branch_head.pt" $OUT | head -1); [ -n "$B" ] || B=runs/s1-heads-337761/branch_head.pt
grep -E "best epoch|cross-fitted" $OUT | head -2
python /tmp/combo3.py runs/s1-heads-337761/branch_head.holdout_probs_all_epochs.npy
D=runs/ft40-s1r-memorise-337666/train/weights/best.pt
python models/yolo/branch_head_predict.py --detector $D --branch $B --images data/course-data/val/images.json --split-root data/course-data/val --out runs/s1-heads-337761/val_probs.json --device cpu
Z=nonsubmissions/241026/s1-memorise-337666
python - <<PY
import json, zipfile, sys
import numpy as np
sys.path.insert(0, 'models/yolo')
from classifier_combo_study import apply_size_rule, size_allowed
images = json.load(open('data/course-data/val/images.json'))['images']
probs = json.load(open('runs/s1-heads-337761/val_probs.json'))
ids = [im['id'] for im in images]
P = apply_size_rule(np.array([probs[str(i)] for i in ids]), [(im['width'], im['height']) for im in images], size_allowed())
labels = P.argmax(1)
src = zipfile.ZipFile('$Z/s1-memorise-337666-rounded-hedged.zip')
old = dict(l.split(',') for l in src.read('classification.csv').decode().strip().splitlines()[1:])
csv = 'image_id,label\n' + ''.join(f'{i},{int(k)}\n' for i, k in zip(ids, labels))
with zipfile.ZipFile('$Z/s1-memorise-337666-rounded-hedged-s1cls.zip', 'w', zipfile.ZIP_DEFLATED) as out:
    out.writestr('detection.json', src.read('detection.json')); out.writestr('classification.csv', csv)
print('changed labels vs the copied soup labels:', sum(old[str(i)] != str(int(k)) for i, k in zip(ids, labels)), 'of', len(ids))
PY
echo S1_BRANCH_ZIP_READY
