#!/bin/bash
# Goal Δm 70 (user, 28 Oct ~22:20): does the flip ladder keep gaining? On Codabench the all-data refits went 59.80 (soup)
# -> 59.75 (+9 epochs) -> 59.93 (+19); refit 3 (+29) is training. Holdout analog of the next rung: the holdout refit
# 340514 (338727 + 10) plus 10 more epochs of the same recipe, scored by holdout group against 338727 and 340514.
#SBATCH --job-name=sar-fliprefit-test2
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --time=10:00:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-frt2-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
OUT=~/sbatch-staging/fliprefit_eval.out
CLS=runs/third-stage-331789-package/image_classifier.pt
TAG=ft40-s1r-fliprefit2-$SLURM_JOB_ID
echo "== $(date '+%m-%d %H:%M') holdout flip refit 2 $TAG started on $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $OUT
python -u models/yolo/finetune_detector.py --tag $TAG --source runs/ft40-s1r-fliprefit-340514/train/weights/best.pt \
    --classifier $CLS --no-augment --lr0 0.001 --epochs 10 --stem-stride 1 --batch 16 --tal-beta 8 --dataset-flips \
    || echo "holdout flip refit 2 FAILED" >> $OUT
[ -f results/$TAG/one2many/predictions_val.json ] && {
  echo "== $(date '+%m-%d %H:%M') holdout flip refit 2 $TAG" >> $OUT
  python ~/sbatch-staging/group_runs.py ft40-s1r-flipmem-338727 ft40-s1r-fliprefit-340514 $TAG 2>&1 | grep -v -i warn >> $OUT; }
echo FLIPREFIT_TEST2_DONE >> $OUT
