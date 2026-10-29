#!/bin/bash
# Two-head classifier for a packaged all-data ZIP: train the conv branch on all data, predict val, rewrite the ZIP's
# classification.csv. Usage: sbatch twohead.sbatch <detector.pt> <name> <zip> <base probabilities json>
#SBATCH --job-name=sar-2head
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/2head-$SLURM_JOB_ID PYTHONUNBUFFERED=1
DET=$1; NAME=$2; ZIP=$3; MEAN=$4; OUT=runs/soup-bp4/$NAME-conv
python -u models/yolo/conv_head_study.py --detector $DET --output $OUT/conv_head.pt --epochs 25 --all-data
# check: the prediction step on the holdout must match the probabilities saved by training (same branch, same images)
python -u models/yolo/conv_head_predict.py --detector $DET --branch $OUT/conv_head.pt --images results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json --split-root data/course-data/train --out $OUT/holdout_check.json
python -u - $OUT <<PY
import json, sys
import numpy as np
from pathlib import Path
out = Path(sys.argv[1])
split = json.loads(Path('runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json').read_text())
ids = list(map(int, split['val_image_ids']))
a = np.load(out / 'conv_head.holdout_probs.npy'); b = json.loads((out / 'holdout_check.json').read_text())
b = np.array([b[str(i)] for i in ids])
print(f'CHECK prediction vs training: same top class on {100 * (a.argmax(1) == b.argmax(1)).mean():.1f}% of holdout images, '
      f'mean |prob difference| {np.abs(a - b).mean():.4f}')
PY
python -u models/yolo/conv_head_predict.py --detector $DET --branch $OUT/conv_head.pt --images data/course-data/val/images.json --split-root data/course-data/val --out $OUT/val_probs.json
python -u models/yolo/combine_classification.py --zip $ZIP --mean-probs $MEAN --conv-probs $OUT/val_probs.json --out ${ZIP%.zip}-2head.zip
echo "TWOHEAD_READY ${ZIP%.zip}-2head.zip"
