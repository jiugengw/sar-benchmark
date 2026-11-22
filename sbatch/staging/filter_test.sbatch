#!/bin/bash
# 29 Oct: a FIXED smoothing filter on every input (training and prediction). Twins lose
# 85.3 vs 89.6 (copy boxes on twin labels) mostly to looser boxes (twin_vs_copy.py); the same filter on both images
# halves their difference (twin_fixed_filter.py). Test = 340514 + 10 epochs of its recipe with SAR_INPUT_FILTER=$FILTER.
# Rule (set before the result): vs 340514, official-mix >= +0.15 and twins >= +1.0.
#SBATCH --job-name=sar-filter-test
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --time=10:00:00
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-ft-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo SAR_INPUT_FILTER=$FILTER
OUT=~/sbatch-staging/filter_eval.out; TAG=ft40-s1r-$FILTER-$SLURM_JOB_ID
echo "== $(date '+%m-%d %H:%M') input-filter test $TAG started on $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $OUT
python -u models/yolo/finetune_detector.py --tag $TAG --source runs/ft40-s1r-fliprefit-340514/train/weights/best.pt \
    --classifier runs/third-stage-331789-package/image_classifier.pt --no-augment --lr0 0.001 --epochs 10 --stem-stride 1 \
    --batch 16 --tal-beta 8 --dataset-flips || echo "input-filter test $TAG FAILED" >> $OUT
[ -f results/$TAG/one2many/predictions_val.json ] && {
  echo "== $(date '+%m-%d %H:%M') input-filter test $TAG (filter at prediction: $(grep -o '"input_filter": "[a-z0-9]*"' results/$TAG/one2many/metrics.json | head -1); reference 341749 = the same 10 epochs with twin jitter)" >> $OUT
  python ~/sbatch-staging/group_runs.py ft40-s1r-flipmem-338727 ft40-s1r-fliprefit-340514 ft40-s1r-twinjit-341749 $TAG 2>&1 | grep -v -i warn >> $OUT; }
echo "FILTER_TEST_DONE $FILTER" >> $OUT
