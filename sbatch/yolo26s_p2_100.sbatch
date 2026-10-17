#!/bin/bash
#SBATCH --job-name=sar-yolo26s-p2
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/yolo26s-p2-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

RUN_ROOT="runs/yolo26s-100-p2-${SLURM_JOB_ID}"
RESULT_ROOT="results/yolo26s-100-p2-${SLURM_JOB_ID}/seed_0"
mkdir -p "$RUN_ROOT" "$RESULT_ROOT"

python -u models/yolo/train_yolo26s_p2.py \
    --weights pretrained/yolo26s.pt \
    --train-output "$RUN_ROOT" \
    --epochs 100 --batch 16 --workers 4

python -u models/yolo/train_image_head.py \
    --detector-checkpoint "$RUN_ROOT/train/weights/best.pt" \
    --data data/course-data \
    --dataset-root data/yolo11s-sar-512 \
    --split runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json \
    --output "$RUN_ROOT/image_classifier.pt" \
    --epochs 100 --batch 32 --workers 4

python -u models/yolo/run_experiment.py \
    --data data/course-data \
    --dataset-yaml data/yolo11s-sar-512/dataset.yaml \
    --dataset-root data/yolo11s-sar-512 \
    --split runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json \
    --baseline-results results/tm-48baseline-326894/terramind_pretrained_moelora/seed_0 \
    --baseline-checkpoint checkpoints/tm-48baseline-326894/terramind_pretrained_moelora/seed_0/model.pt \
    --weights pretrained/yolo26s.pt \
    --train-output "$RUN_ROOT" \
    --results-output "$RESULT_ROOT" \
    --epochs 100 --batch 16 --workers 4 \
    --classifier-checkpoint "$RUN_ROOT/image_classifier.pt" \
    --model-label "YOLO26s-100-P2" \
    --eval-only

python reports/training-comparison/build_comparison.py
