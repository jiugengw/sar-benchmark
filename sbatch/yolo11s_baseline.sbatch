#!/bin/bash
#SBATCH --job-name=sar-yolo11s-pilot
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/yolo11s-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/yolo11s-matplotlib-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

JOB_ROOT="runs/yolo11s-baseline-${SLURM_JOB_ID}"
RESULT_ROOT="results/yolo11s-baseline-${SLURM_JOB_ID}/seed_0"
mkdir -p "$JOB_ROOT" "$RESULT_ROOT"

python -u models/yolo/run_experiment.py \
    --data data/course-data \
    --dataset-yaml data/yolo11s-sar-512/dataset.yaml \
    --dataset-root data/yolo11s-sar-512 \
    --split runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json \
    --baseline-results results/tm-48baseline-326894/terramind_pretrained_moelora/seed_0 \
    --baseline-checkpoint checkpoints/tm-48baseline-326894/terramind_pretrained_moelora/seed_0/model.pt \
    --weights pretrained/yolo11s.pt \
    --train-output "$JOB_ROOT" \
    --results-output "$RESULT_ROOT" \
    --epochs 24 --batch 16 --workers 4

# Refresh the selectable report with the completed YOLO run. A report-rendering
# hiccup should not mark a completed training job as failed.
