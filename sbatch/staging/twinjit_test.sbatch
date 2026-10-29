#!/bin/bash
# Goal Δm 70 / Phase 1 (28 Oct ~23:55): twin-matched jitter in memorise. Holdout twins score 84.8 against 98.3 on their
# training copies; they are pixel-aligned (median shift 0.02 px) and differ by gain 0.8-1.1, offset and smoothing
# (Lee / Gaussian 0.5-2). TwinJitter (p 0.5) adds those; test = 340514 + 10 epochs of its recipe with it.
#SBATCH --job-name=sar-twinjit-test
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --time=10:00:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-tj-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
OUT=~/sbatch-staging/twinjit_eval.out; TAG=ft40-s1r-twinjit-$SLURM_JOB_ID
echo "== $(date '+%m-%d %H:%M') twin-jitter test $TAG started on $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $OUT
python -u models/yolo/finetune_detector.py --tag $TAG --source runs/ft40-s1r-fliprefit-340514/train/weights/best.pt \
    --classifier runs/third-stage-331789-package/image_classifier.pt --no-augment --lr0 0.001 --epochs 10 --stem-stride 1 \
    --batch 16 --tal-beta 8 --dataset-flips --twin-jitter 0.5 || echo "twin-jitter test FAILED" >> $OUT
[ -f results/$TAG/one2many/predictions_val.json ] && {
  echo "== $(date '+%m-%d %H:%M') twin-jitter test $TAG (control: 340514 = the same start without the 10 epochs; the plain rung 338727 -> 340514 gave official-mix +0.14, twins +0.5)" >> $OUT
  python ~/sbatch-staging/group_runs.py ft40-s1r-flipmem-338727 ft40-s1r-fliprefit-340514 $TAG 2>&1 | grep -v -i warn >> $OUT; }
echo TWINJIT_TEST_DONE >> $OUT
