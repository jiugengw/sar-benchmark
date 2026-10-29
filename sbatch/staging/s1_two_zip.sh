#!/bin/bash
# Single-model ZIP: M stride-1 boxes (rounded + hedged, from the uploaded s1cls ZIP) + M stride-1 two-head image labels
# (0.5 branch + 0.5 conv head, both on M stride-1 features, size rule). CPU on the login node.
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
H=runs/s1-heads-337761; D=runs/ft40-s1r-memorise-337666/train/weights/best.pt
python models/yolo/conv_head_predict.py --detector $D --branch $H/conv_head.pt --images data/course-data/val/images.json --split-root data/course-data/val --out $H/val_probs_conv.json --device cpu
python - <<PY
import json, zipfile, sys
import numpy as np
sys.path.insert(0, 'models/yolo')
from classifier_combo_study import apply_size_rule, size_allowed
images = json.load(open('data/course-data/val/images.json'))['images']; ids = [im['id'] for im in images]
b = json.load(open('$H/val_probs.json')); c = json.load(open('$H/val_probs_conv.json'))
P = apply_size_rule(0.5 * np.array([b[str(i)] for i in ids]) + 0.5 * np.array([c[str(i)] for i in ids]), [(im['width'], im['height']) for im in images], size_allowed())
labels = P.argmax(1)
src = zipfile.ZipFile('/tmp/s1cls.zip')
old = dict(l.split(',') for l in src.read('classification.csv').decode().strip().splitlines()[1:])
with zipfile.ZipFile('nonsubmissions/251026/s1-memorise-337666-rounded-hedged-s1two.zip', 'w', zipfile.ZIP_DEFLATED) as out:
    out.writestr('detection.json', src.read('detection.json'))
    out.writestr('classification.csv', 'image_id,label\n' + ''.join(f'{i},{int(k)}\n' for i, k in zip(ids, labels)))
print('labels changed vs the s1cls upload:', sum(old[str(i)] != str(int(k)) for i, k in zip(ids, labels)), 'of', len(ids))
PY
echo S1TWO_READY
