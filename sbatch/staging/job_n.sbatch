#!/bin/bash
# job_n: one generalisation direction on the training split, judged on NEW scenes with job_g.py against the plain
# stride-1 memorise 337666 (same start ft40-s1r-third-337119, 40 epochs, lr 0.002, no augmentation, beta 8, batch 16).
# 31 Oct, user: "try stuff in diff direction, and make sure generalise for phase 2 ... good on new unseen images".
#   V=freeze11 sbatch --job-name=job-n2 --export=ALL ~/sbatch-staging/job_n.sbatch
#   V=flipsel   the final model's recipe (class-selective flip memorise, 20 epochs over the 4x list as 338727): does
#               the recipe we submit lose on new scenes against plain memorise?
#   V=freeze11  backbone (layers 0-10) kept fixed during memorise: only the neck and head sharpen
#   V=swaconst  constant lr 0.002 with a checkpoint every 2 epochs, then the average of the second half (SWA)
#   V=dedupe    one image per same-class twin group (repeated scenes count once)
#   V=wd5e3     weight decay 0.005 instead of 0.0005
#SBATCH --job-name=job-n
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-n-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 SAR_INPUT_FILTER=none
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader); variant ${V:?set V}"
S="--source runs/ft40-s1r-third-337119/train/weights/best.pt --classifier runs/third-stage-331789-package/image_classifier.pt
   --lr0 0.002 --no-augment --stem-stride 1 --tal-beta 8 --batch 16"
TAG=ft40-s1r-$V-$SLURM_JOB_ID
SPEC=runs/$TAG.spec.json
case $V in
  flipsel) python -u models/yolo/finetune_detector.py --tag $TAG $S --epochs 20 --dataset-flips \
             --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine
           echo "{\"score\": [\"ft40-s1r-memorise-337666\", \"ft40-s1r-flipmem-338727\", \"$TAG\"]}" > $SPEC ;;
  freeze11) python -u models/yolo/finetune_detector.py --tag $TAG $S --epochs 40 --freeze 11
           echo "{\"score\": [\"ft40-s1r-memorise-337666\", \"$TAG\"]}" > $SPEC ;;
  swaconst) python -u models/yolo/finetune_detector.py --tag $TAG $S --epochs 40 --constant-lr --save-period 2
           python3 - runs/$TAG/train/weights $TAG > $SPEC <<'PY'
import json, re, sys
from pathlib import Path
w, tag = Path(sys.argv[1]), sys.argv[2]
half = sorted((int(re.findall(r'\d+', p.stem)[0]), str(p)) for p in w.glob('epoch*.pt'))
parts = [[p, 1] for e, p in half if e >= 20] + [[str(w / 'last.pt'), 1]]
print(json.dumps({'build': {f'{tag}-swa': parts}, 'score': ['ft40-s1r-memorise-337666', tag, f'{tag}-swa']}))
PY
           ;;
  dedupe) python -u models/yolo/finetune_detector.py --tag $TAG $S --epochs 40 --dedupe-twins
           echo "{\"score\": [\"ft40-s1r-memorise-337666\", \"$TAG\"]}" > $SPEC ;;
  wd5e3) python -u models/yolo/finetune_detector.py --tag $TAG $S --epochs 40 --weight-decay 0.005
           echo "{\"score\": [\"ft40-s1r-memorise-337666\", \"$TAG\"]}" > $SPEC ;;
  *) echo "unknown variant $V"; exit 2 ;;
esac
python -u ~/sbatch-staging/job_g.py $SPEC
echo "JOB_N_DONE $V $(date '+%m-%d %H:%M')"
