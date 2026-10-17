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
export PYTHONPATH="${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$RUN_ROOT" "$RESULT_ROOT"

python -u models/yolo/train_yolo26s_p2.py \
    --scale "$MODEL_SCALE" \
    --weights "pretrained/${MODEL_NAME}.pt" \
    --train-output "$RUN_ROOT" \
    --epochs 200 --batch 16 --workers 4 \
    --flipud 0.5 --close-mosaic 20

# Image classifier on the frozen detector: the 1024-neuron recipe of the best submission.
python -u models/yolo/train_image_head.py \
    --detector-checkpoint "$RUN_ROOT/train/weights/best.pt" \
    --data data/course-data \
    --dataset-root data/yolo11s-sar-512 \
    --split runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json \
    --output "$RUN_ROOT/image_classifier.pt" \
    --feature-pooling spatial --hidden-dim 1024 --dropout 0.35 --learning-rate 0.0005 \
    --weight-decay 0.001 --train-flips --head-batch 256 \
    --epochs 240 --batch 32 --workers 4

# Score both prediction heads through the exact submission code, on the full and clean holdout.
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
for BRANCH in one2many one2one; do
    python - "$RUN_ROOT/train/weights/best.pt" "$RUN_ROOT/image_classifier.pt" "$BRANCH" "$RESULT_ROOT/../$BRANCH" <<'PY'
import sys
from pathlib import Path
from models.yolo.deadline_recovery import evaluate_export
evaluate_export(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
PY
    cp results/yolo26m-100-p2-328339/seed_0/instances_val_holdout.json "$RESULT_ROOT/../$BRANCH/"
    python -m sarbench.twins score "$RESULT_ROOT/../$BRANCH" || true
done

# Keep the per-epoch history in git (runs/ is not committed) for the curve graphs.
CURVES="results/training-curves/${MODEL_NAME}-200-p2-${SLURM_JOB_ID}"
mkdir -p "$CURVES"
cp "$RUN_ROOT/train/results.csv" "$CURVES/training.csv"
cp "$RUN_ROOT/train/args.yaml" "$CURVES/args.yaml"

PACKAGE_DIR="nonsubmissions/$(date +%d%m%y)/${MODEL_NAME}-200-p2-${SLURM_JOB_ID}"
PACKAGE="$PACKAGE_DIR/${MODEL_NAME}_200_p2_${SLURM_JOB_ID}"
mkdir -p "$PACKAGE_DIR"
python -u models/yolo/make_submission.py \
    --weights "$RUN_ROOT/train/weights/best.pt" \
    --classifier-checkpoint "$RUN_ROOT/image_classifier.pt" \
    --detection-branch one2many \
    --data data/course-data \
    --classes data/yolo11s-sar-512/conversion_manifest.json \
    --out "$PACKAGE.zip" \
    --batch 16 --workers 4
cp "$RESULT_ROOT/../one2many/metrics.json" "$PACKAGE_DIR/holdout.metrics.json"
printf '{"zip": "%s", "histories": [{"type": "ultralytics", "path": "%s/training.csv", "label": "YOLO26%s-P2, 200 epochs, vertical flips"}]}\n' \
    "$(basename "$PACKAGE").zip" "$CURVES" "$MODEL_SCALE" > "$PACKAGE.run.json"
echo "Package: $PACKAGE.zip  (holdout scores: $RESULT_ROOT/../one2many/metrics.json)"
