#!/bin/bash
#SBATCH --job-name=sar-s1fssoup-heads
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/s1sh-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
DET=runs/soup-s1fs/alldata-s1fs-six.pt; H=runs/soup-s1fs/heads-$SLURM_JOB_ID
python -u models/yolo/branch_head_study.py --detector $DET --output $H/branch_head.pt --epochs 20 --batch 16 --all-data
python -u models/yolo/conv_head_study.py --detector $DET --output $H/conv_head.pt --batch 16 --epochs 30 --all-data
python -u models/yolo/branch_head_predict.py --detector $DET --branch $H/branch_head.pt --images data/course-data/val/images.json --split-root data/course-data/val --out $H/val_probs_branch.json
python -u models/yolo/conv_head_predict.py --detector $DET --branch $H/conv_head.pt --images data/course-data/val/images.json --split-root data/course-data/val --out $H/val_probs_conv.json
python - <<PY
import json, zipfile, sys
import numpy as np
sys.path.insert(0, 'models/yolo')
from classifier_combo_study import apply_size_rule, size_allowed
images = json.load(open('data/course-data/val/images.json'))['images']; ids = [im['id'] for im in images]
b = json.load(open('$H/val_probs_branch.json')); c = json.load(open('$H/val_probs_conv.json'))
P = apply_size_rule(0.5 * np.array([b[str(i)] for i in ids]) + 0.5 * np.array([c[str(i)] for i in ids]), [(im['width'], im['height']) for im in images], size_allowed())
labels = P.argmax(1)
src = zipfile.ZipFile('nonsubmissions/271026/alldata-s1fs-soup6/alldata-s1fs-soup6-rounded-hedged.zip')
old = dict(l.split(',') for l in src.read('classification.csv').decode().strip().splitlines()[1:])
target = 'nonsubmissions/271026/alldata-s1fs-soup6/alldata-s1fs-soup6-rounded-hedged.zip'.replace('-rounded-hedged.zip', '-rounded-hedged-s1two.zip')
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as out:
    out.writestr('detection.json', src.read('detection.json'))
    out.writestr('classification.csv', 'image_id,label\n' + ''.join(f'{i},{int(k)}\n' for i, k in zip(ids, labels)))
print('S1FS_SOUP_TWOHEAD_ZIP', target, 'labels changed vs the mean-head ZIP:', sum(old[str(i)] != str(int(k)) for i, k in zip(ids, labels)))
PY
