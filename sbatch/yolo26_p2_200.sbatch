#!/bin/bash
#SBATCH --job-name=sar-yolo26-p2-200
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_SCALE="${MODEL_SCALE:?Submit one scale with --export=MODEL_SCALE=n|s|m|l}"
case "$MODEL_SCALE" in
    n|s|m|l) ;;
    *) echo "Unsupported YOLO26 scale: $MODEL_SCALE" >&2; exit 2 ;;
esac
MODEL_NAME="yolo26${MODEL_SCALE}"
RUN_ROOT="runs/${MODEL_NAME}-200-p2-${SLURM_JOB_ID}"
RESULT_ROOT="results/${MODEL_NAME}-200-p2-${SLURM_JOB_ID}/seed_0"
WEIGHTS_PATH="pretrained/${MODEL_NAME}.pt"
if [[ "$MODEL_SCALE" == n ]]; then
    WEIGHTS_PATH="${RUN_ROOT}/pretrained_yolo26n.pt"
fi
export MPLCONFIGDIR="/tmp/${MODEL_NAME}-p2-200-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/yolo_baseline${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$RUN_ROOT" "$RESULT_ROOT"

python -u yolo_baseline/train_yolo26s_p2.py \
    --scale "$MODEL_SCALE" \
    --weights "pretrained/${MODEL_NAME}.pt" \
    --train-output "$RUN_ROOT" \
    --epochs 200 --batch 16 --workers 4

python -u yolo_baseline/train_image_head.py \
    --detector-checkpoint "$RUN_ROOT/train/weights/best.pt" \
    --data data/course-data \
    --dataset-root data/yolo11s-sar-512 \
    --split runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json \
    --output "$RUN_ROOT/image_classifier.pt" \
    --epochs 100 --batch 32 --workers 4

python -u yolo_baseline/run_experiment.py \
    --data data/course-data \
    --dataset-yaml data/yolo11s-sar-512/dataset.yaml \
    --dataset-root data/yolo11s-sar-512 \
    --split runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json \
    --baseline-results results/tm-48baseline-326894/terramind_pretrained_moelora/seed_0 \
    --baseline-checkpoint checkpoints/tm-48baseline-326894/terramind_pretrained_moelora/seed_0/model.pt \
    --weights "$WEIGHTS_PATH" \
    --train-output "$RUN_ROOT" \
    --results-output "$RESULT_ROOT" \
    --epochs 200 --batch 16 --workers 4 \
    --classifier-checkpoint "$RUN_ROOT/image_classifier.pt" \
    --model-label "YOLO26${MODEL_SCALE}-200-P2" \
    --eval-only

python reports/training-comparison/build_comparison.py

PACKAGE_DIR="nonsubmissions/161026/tests-no-submit/${MODEL_NAME}-200-p2-${SLURM_JOB_ID}"
mkdir -p "$PACKAGE_DIR/scoring_result"
touch "$PACKAGE_DIR/scoring_result/.gitkeep"
python -u yolo_baseline/make_submission.py \
    --weights "$RUN_ROOT/train/weights/best.pt" \
    --classifier-checkpoint "$RUN_ROOT/image_classifier.pt" \
    --box-class-policy native_detector_classes \
    --data data/course-data \
    --classes data/yolo11s-sar-512/conversion_manifest.json \
    --out "$PACKAGE_DIR/${MODEL_NAME}_200_p2_${SLURM_JOB_ID}.zip" \
    --batch 16 --workers 4
cp "$RESULT_ROOT/metrics.json" "$PACKAGE_DIR/${MODEL_NAME}_200_p2_${SLURM_JOB_ID}.metrics.json"
