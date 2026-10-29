#!/bin/bash
# Two holdout-split tests in ONE job (the free slot while the flip soup finishes), each scored by holdout group:
# 1. Speckle soups, no training: average the flip-memorise test (338727, clean pixels) with its speckle versions
#    (338730 constant speckle: near-copies +3.5 but flipped copies -6.2 and new -2.2; 338803 annealed). A weight
#    average may keep the clean model's exact-pixel memory and the speckled model's re-capture recognition.
# 2. Twin test: 338727's recipe plus brightness (+-20%, --photometric) and sharpness jitter (--sharpness-prob 0.5),
#    aimed at twins (same capture as a training image, other brightness/sharpness; 219 holdout images at 84.8 mAP).
#SBATCH --job-name=sar-spktwin
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --time=16:00:00
#SBATCH --output=runs/sar-spktwin-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-spktwin-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
OUT=~/sbatch-staging/spktwin_eval.out
CLS=runs/third-stage-331789-package/image_classifier.pt
BASE="ft40-s1r-memorise-337666 ft40-s1r-flipmem-338727 ft40-s1r-flipspk-338730 ft40-s1r-flipanneal-338803"
SOUPS="soup-s1r-flip-spk soup-s1r-2flip-spk soup-s1r-flip-anneal soup-s1r-flip-spk-anneal"

echo "== $(date '+%m-%d %H:%M') speckle soups (job $SLURM_JOB_ID, $(nvidia-smi --query-gpu=name --format=csv,noheader))" >> $OUT
python - "$CLS" <<'PY'
import sys
from pathlib import Path
from swa_study import average
from deadline_recovery import evaluate_export
runs = {'flip': 'ft40-s1r-flipmem-338727', 'spk': 'ft40-s1r-flipspk-338730', 'anneal': 'ft40-s1r-flipanneal-338803'}
soups = {'soup-s1r-flip-spk': ['flip', 'spk'], 'soup-s1r-2flip-spk': ['flip', 'flip', 'spk'],
         'soup-s1r-flip-anneal': ['flip', 'anneal'], 'soup-s1r-flip-spk-anneal': ['flip', 'spk', 'anneal']}
for tag, parts in soups.items():  # best.pt, as each test was scored
    out = Path('results') / tag
    out.mkdir(parents=True, exist_ok=True)
    weights = average([Path(f'runs/{runs[p]}/train/weights/best.pt') for p in parts], out / 'averaged.pt')
    evaluate_export(weights, Path(sys.argv[1]), 'one2many', out / 'one2many')
    print('SOUP_SCORED', tag, flush=True)
PY
python ~/sbatch-staging/group_runs.py $BASE $SOUPS 2>&1 | grep -v -i warn >> $OUT

TAG=ft40-s1r-flipjit-$SLURM_JOB_ID
echo "== $(date '+%m-%d %H:%M') twin test $TAG started" >> $OUT
python -u models/yolo/finetune_detector.py --tag $TAG --source runs/ft40-s1r-third-337119/train/weights/best.pt \
    --classifier $CLS --no-augment --lr0 0.002 --epochs 20 --stem-stride 1 --batch 16 --tal-beta 8 --dataset-flips \
    --photometric --sharpness-prob 0.5 || echo "twin test FAILED" >> $OUT
[ -f results/$TAG/one2many/predictions_val.json ] && {
  echo "== $(date '+%m-%d %H:%M') twin test $TAG" >> $OUT
  python ~/sbatch-staging/group_runs.py $BASE $SOUPS $TAG 2>&1 | grep -v -i warn >> $OUT; }
echo SPKTWIN_DONE >> $OUT
