#!/bin/bash
# Long shot for official mAP 60 (28 Oct): are flipped views under-memorised? Exact flipped mirrors in the clean holdout
# score 91.1 with the flip test (338727) against ~98 for as-is copies (airport 85, harbor 89): as-is views were trained
# through every stage, flipped views only in the 20 memorise epochs. Test: 338727 plus 10 more epochs of its own recipe
# (all classes flipped, beta 8) at lr 0.001, scored by holdout group against 338727.
#SBATCH --job-name=sar-fliprefit-test
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
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-frt-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
OUT=~/sbatch-staging/fliprefit_eval.out
CLS=runs/third-stage-331789-package/image_classifier.pt
TAG=ft40-s1r-fliprefit-$SLURM_JOB_ID
echo "== $(date '+%m-%d %H:%M') flip refit test $TAG started on $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $OUT
python -u models/yolo/finetune_detector.py --tag $TAG --source runs/ft40-s1r-flipmem-338727/train/weights/best.pt \
    --classifier $CLS --no-augment --lr0 0.001 --epochs 10 --stem-stride 1 --batch 16 --tal-beta 8 --dataset-flips \
    || echo "flip refit test FAILED" >> $OUT
[ -f results/$TAG/one2many/predictions_val.json ] && {
  echo "== $(date '+%m-%d %H:%M') flip refit test $TAG" >> $OUT
  python ~/sbatch-staging/group_runs.py ft40-s1r-flipmem-338727 $TAG 2>&1 | grep -v -i warn >> $OUT
  python ~/sbatch-staging/flip_exact_ap.py results/$TAG/one2many 2>&1 | grep -v -i warn >> $OUT; }
echo FLIPREFIT_TEST_DONE >> $OUT
