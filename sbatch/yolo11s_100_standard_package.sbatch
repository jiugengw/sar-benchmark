#!/bin/bash
#SBATCH --job-name=sar-yolo11s-std-pack
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/yolo11s-standard-pack-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

PACKAGE_DIR="nonsubmissions/161026/yolo11s-100-standard-328063"
mkdir -p "$PACKAGE_DIR/scoring_result"
touch "$PACKAGE_DIR/scoring_result/.gitkeep"

python -u models/yolo/make_submission.py \
    --weights runs/yolo11s-100-standard-328063/train/weights/best.pt \
    --classifier-checkpoint runs/yolo11s-100-standard-328063/image_classifier.pt \
    --box-class-policy native_detector_classes \
    --data data/course-data \
    --classes data/yolo11s-sar-512/conversion_manifest.json \
    --out "$PACKAGE_DIR/yolo11s_100_standard_328063.zip" \
    --batch 16 --workers 4

cp results/yolo11s-100-standard-328063/seed_0/metrics.json \
    "$PACKAGE_DIR/yolo11s_100_standard_328063.metrics.json"
