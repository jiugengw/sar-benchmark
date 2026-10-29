#!/bin/bash
# PRIVATE diagnostic, never submitted, not in the repo: our recipe (the chain's plain stage: from the 100-epoch DOTA
# model, lr 0.002, usual augmentation) trained on the course training set PLUS every M4-SAR train/val-split image
# not protected (holdout / val scenes, near duplicates, twins, course copies), then scored on M4-SAR's unseen
# single-class test images and our holdout, next to the course-only plain stage (chain-dota-330938-plain).
#SBATCH --job-name=sar-m4full
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=/common/scratch/users/j/jgwong.2023/private-val/%x-%j.out
set -euo pipefail
cd /common/scratch/users/j/jgwong.2023/sar-benchmark
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/sar-m4full-${SLURM_JOB_ID}"
export PYTHONPATH="$PWD:$PWD/models/yolo"
EPOCHS="${1:-30}"
TAG="private-m4full-plain-${SLURM_JOB_ID}"
python -u ../private-val/m4_finetune.py --extra 1000000 --skip-m4-test --tag "$TAG" \
    --source runs/yolo26m-p2-dota-verified-330705/train/weights/best.pt \
    --classifier runs/yolo26m-p2-dota-verified-330705/image_classifier.pt --seed 0 --epochs "$EPOCHS" --lr0 0.002 \
    --save-period 0
python -u -c "
from pathlib import Path
from models.yolo.deadline_recovery import evaluate_export
evaluate_export(Path('runs/chain-dota-330938-plain/train/weights/best.pt'), Path('runs/yolo26m-p2-dota-verified-330705/image_classifier.pt'), 'one2many', Path('results/private-m4full-control-plain-330938/one2many'))"
python -u ../private-val/m4_test_eval.py "runs/$TAG/train/weights/best.pt" runs/chain-dota-330938-plain/train/weights/best.pt
echo "M4FULL_COMPLETE $TAG"
