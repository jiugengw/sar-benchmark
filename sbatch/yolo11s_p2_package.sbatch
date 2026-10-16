#!/bin/bash
#SBATCH --job-name=sar-yolo11s-p2-pack
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
export MPLCONFIGDIR="/tmp/yolo11s-p2-pack-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}/yolo_baseline${PYTHONPATH:+:${PYTHONPATH}}"

P2_PACKAGE_DIR="submissions/161026/tests-no-submit/yolo11s-100-p2-328058"
STANDARD_PACKAGE_DIR="submissions/161026/tests-no-submit/yolo11s-100-standard-328063"
mkdir -p "$P2_PACKAGE_DIR/scoring_result" "$STANDARD_PACKAGE_DIR/scoring_result"
touch "$P2_PACKAGE_DIR/scoring_result/.gitkeep" "$STANDARD_PACKAGE_DIR/scoring_result/.gitkeep"

python -u yolo_baseline/make_submission.py \
    --weights runs/yolo11s-100-p2-328058/train/weights/best.pt \
    --classifier-checkpoint runs/yolo11s-100-p2-328058/image_classifier.pt \
    --box-class-policy native_detector_classes \
    --data data/course-data \
    --classes data/yolo11s-sar-512/conversion_manifest.json \
    --out "$P2_PACKAGE_DIR/yolo11s_100_p2_328058.zip" \
    --batch 16 --workers 4

cp results/yolo11s-100-p2-328058/seed_0/metrics.json \
    "$P2_PACKAGE_DIR/yolo11s_100_p2_328058.metrics.json"

python -u yolo_baseline/make_submission.py \
    --weights runs/yolo11s-100-standard-328063/train/weights/best.pt \
    --classifier-checkpoint runs/yolo11s-100-standard-328063/image_classifier.pt \
    --box-class-policy native_detector_classes \
    --data data/course-data \
    --classes data/yolo11s-sar-512/conversion_manifest.json \
    --out "$STANDARD_PACKAGE_DIR/yolo11s_100_standard_328063.zip" \
    --batch 16 --workers 4

cp results/yolo11s-100-standard-328063/seed_0/metrics.json \
    "$STANDARD_PACKAGE_DIR/yolo11s_100_standard_328063.metrics.json"
