#!/bin/bash
# Standalone Phase 2 E9: YOLO11n-P2 detector, image classifier, evaluation, and ZIP.
# Submit from the project root with: sbatch sbatch/phase2_e9_yolo11n.sbatch
# This runs E9 independently of the dropped, time-intensive E8 stage.
#SBATCH --job-name=sar-phase2-e9-yolo11n
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
export MPLCONFIGDIR="/tmp/sar-phase2-e9-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/yolo_baseline${PYTHONPATH:+:${PYTHONPATH}}"

DATA="data/course-data"
DATASET="data/yolo11s-sar-512"
SPLIT="runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json"
BASELINE_RESULTS="results/tm-48baseline-326894/terramind_pretrained_moelora/seed_0"
BASELINE_CHECKPOINT="checkpoints/tm-48baseline-326894/terramind_pretrained_moelora/seed_0/model.pt"
MANIFEST="$DATASET/conversion_manifest.json"
RUN_ROOT="runs/yolo11n-100-p2-${SLURM_JOB_ID}"
RESULT_ROOT="results/yolo11n-100-p2-${SLURM_JOB_ID}/seed_0"
PACKAGE_DIR="nonsubmissions/sar-experiments/E9-${SLURM_JOB_ID}"

for input in \
    pretrained/yolo11n.pt \
    "$DATASET/dataset.yaml" \
    "$MANIFEST" \
    "$DATA/train/labels.csv" \
    "$SPLIT" \
    "$BASELINE_RESULTS/metrics.json" \
    "$BASELINE_CHECKPOINT"; do
    if [[ ! -s "$input" ]]; then
        echo "Missing E9 input: $input" >&2
        exit 2
    fi
done

python -u yolo_baseline/train_100_epoch.py \
    --scale n --variant p2 --weights pretrained/yolo11n.pt \
    --dataset-yaml "$DATASET/dataset.yaml" --classes "$MANIFEST" \
    --train-output "$RUN_ROOT" --epochs 100 --batch 16 --workers 4

python -u yolo_baseline/train_image_head.py \
    --architecture yolo --detector-checkpoint "$RUN_ROOT/train/weights/best.pt" \
    --data "$DATA" --dataset-root "$DATASET" --split "$SPLIT" \
    --output "$RUN_ROOT/image_classifier.pt" --epochs 100 --batch 32 --workers 4

python -u yolo_baseline/run_experiment.py \
    --data "$DATA" --dataset-yaml "$DATASET/dataset.yaml" \
    --dataset-root "$DATASET" --split "$SPLIT" \
    --baseline-results "$BASELINE_RESULTS" --baseline-checkpoint "$BASELINE_CHECKPOINT" \
    --weights pretrained/yolo11n.pt --architecture yolo \
    --train-output "$RUN_ROOT" --results-output "$RESULT_ROOT" \
    --epochs 100 --batch 16 --workers 4 \
    --classifier-checkpoint "$RUN_ROOT/image_classifier.pt" \
    --model-label "YOLO11n-P2 E9" --eval-only

mkdir -p "$PACKAGE_DIR/scoring_result"
touch "$PACKAGE_DIR/scoring_result/.gitkeep"
python -u yolo_baseline/make_submission.py \
    --weights "$RUN_ROOT/train/weights/best.pt" \
    --classifier-checkpoint "$RUN_ROOT/image_classifier.pt" \
    --box-class-policy native_detector_classes --data "$DATA" --classes "$MANIFEST" \
    --out "$PACKAGE_DIR/E9_${SLURM_JOB_ID}.zip" --batch 16 --workers 4

cp "$RESULT_ROOT/metrics.json" "$PACKAGE_DIR/E9_${SLURM_JOB_ID}.metrics.json"
cp "$RUN_ROOT/compute.json" "$PACKAGE_DIR/E9_${SLURM_JOB_ID}.compute.json"
cp "$RUN_ROOT/benchmark_settings.json" "$PACKAGE_DIR/E9_${SLURM_JOB_ID}.config.json"
cp "$RUN_ROOT/image_classifier.metrics.json" "$PACKAGE_DIR/E9_${SLURM_JOB_ID}.classifier.metrics.json"

echo "E9 finished: $RUN_ROOT"
echo "E9 results: $RESULT_ROOT"
echo "E9 package: $PACKAGE_DIR/E9_${SLURM_JOB_ID}.zip"
