#!/bin/bash
#SBATCH --job-name=sar-yolo-variant
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
export MPLCONFIGDIR="/tmp/${YOLO_MODEL:?Set YOLO_MODEL}-matplotlib-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

case "$YOLO_MODEL" in
    yolo11n|yolo11s|yolo11m|yolo11l|yolo26s|yolo26m|yolo26l) ;;
    *) echo "Unsupported YOLO_MODEL: $YOLO_MODEL" >&2; exit 2 ;;
esac

RUN_ID="${SLURM_JOB_ID}"
RUN_ARGS=()
if [[ "${YOLO_EVAL_ONLY:-0}" == "1" ]]; then
    RUN_ID="${YOLO_TRAIN_JOB_ID:?Set YOLO_TRAIN_JOB_ID for evaluation-only recovery}"
    RUN_ARGS+=(--eval-only)
fi

JOB_ROOT="runs/${YOLO_MODEL}-baseline-${RUN_ID}"
RESULT_ROOT="results/${YOLO_MODEL}-baseline-${RUN_ID}/seed_0"
mkdir -p "$JOB_ROOT" "$RESULT_ROOT"

python -u models/yolo/run_experiment.py \
    --data data/course-data \
    --dataset-yaml data/yolo11s-sar-512/dataset.yaml \
    --dataset-root data/yolo11s-sar-512 \
    --split runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json \
    --baseline-results results/tm-48baseline-326894/terramind_pretrained_moelora/seed_0 \
    --baseline-checkpoint checkpoints/tm-48baseline-326894/terramind_pretrained_moelora/seed_0/model.pt \
    --weights "pretrained/${YOLO_MODEL}.pt" \
    --train-output "$JOB_ROOT" \
    --results-output "$RESULT_ROOT" \
    --epochs 24 --batch 16 --workers 4 "${RUN_ARGS[@]}"

python reports/training-comparison/build_comparison.py || true
