#!/bin/bash
#SBATCH --job-name=sar-phase1-yolo26s-p2
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/sar-phase1-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="models/yolo/experiment_configs/phase1.yaml"
DATA="data/course-data"
DATASET="data/yolo11s-sar-512"
SPLIT="runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json"
BASELINE_RESULTS="results/tm-48baseline-326894/terramind_pretrained_moelora/seed_0"
BASELINE_CHECKPOINT="checkpoints/tm-48baseline-326894/terramind_pretrained_moelora/seed_0/model.pt"
JOB_SUFFIX="${SLURM_JOB_ID}"

evaluate_and_package() {
    local experiment="$1"
    local run_root="$2"
    local result_root="$3"
    local epochs="$4"
    local label="$5"
    mkdir -p "$result_root"

    python -u models/yolo/run_experiment.py \
        --data "$DATA" \
        --dataset-yaml "$DATASET/dataset.yaml" \
        --dataset-root "$DATASET" \
        --split "$SPLIT" \
        --baseline-results "$BASELINE_RESULTS" \
        --baseline-checkpoint "$BASELINE_CHECKPOINT" \
        --weights "$run_root/train/weights/best.pt" \
        --train-output "$run_root" \
        --results-output "$result_root" \
        --epochs "$epochs" --batch 16 --workers 4 \
        --classifier-checkpoint "$run_root/image_classifier.pt" \
        --model-label "$label" --eval-only

    local package_dir="nonsubmissions/sar-experiments/${experiment}-${JOB_SUFFIX}"
    mkdir -p "$package_dir/scoring_result"
    touch "$package_dir/scoring_result/.gitkeep"
    python -u models/yolo/make_submission.py \
        --weights "$run_root/train/weights/best.pt" \
        --classifier-checkpoint "$run_root/image_classifier.pt" \
        --box-class-policy native_detector_classes \
        --data "$DATA" \
        --classes "$DATASET/conversion_manifest.json" \
        --out "$package_dir/${experiment}_${JOB_SUFFIX}.zip" \
        --batch 16 --workers 4
    cp "$result_root/metrics.json" "$package_dir/${experiment}_${JOB_SUFFIX}.metrics.json"
    cp "$run_root/compute.json" "$package_dir/${experiment}_${JOB_SUFFIX}.compute.json"
    cp "$run_root/experiment_config.json" "$package_dir/${experiment}_${JOB_SUFFIX}.config.json"
}

E2_RUN="runs/yolo26s-p2-E2-${JOB_SUFFIX}"
E2_RESULTS="results/yolo26s-p2-E2-${JOB_SUFFIX}/seed_0"
python -u models/yolo/train_joint.py --config "$CONFIG" --experiment E2 --output "$E2_RUN"
evaluate_and_package E2 "$E2_RUN" "$E2_RESULTS" 15 "YOLO26s-P2 E2 joint fine-tuning"

E3_RUN="runs/yolo26s-p2-E3-${JOB_SUFFIX}"
E3_RESULTS="results/yolo26s-p2-E3-${JOB_SUFFIX}/seed_0"
python -u models/yolo/train_joint.py --config "$CONFIG" --experiment E3 --output "$E3_RUN"
evaluate_and_package E3 "$E3_RUN" "$E3_RESULTS" 100 "YOLO26s-P2 E3 joint from initialization"

E4_RUN="runs/yolo26s-p2-E4-${JOB_SUFFIX}"
E4_RESULTS="results/yolo26s-p2-E4-${JOB_SUFFIX}/seed_0"
python -u models/yolo/continue_detector.py --config "$CONFIG" --output "$E4_RUN"
if [[ ! -f "$E4_RUN/skipped.json" ]]; then
    python -u models/yolo/train_image_head.py \
        --detector-checkpoint "$E4_RUN/train/weights/best.pt" \
        --data "$DATA" --dataset-root "$DATASET" --split "$SPLIT" \
        --output "$E4_RUN/image_classifier.pt" --epochs 100 --batch 32 --workers 4
    evaluate_and_package E4 "$E4_RUN" "$E4_RESULTS" 200 "YOLO26s-P2 E4 continued to 200 epochs"
fi

python reports/training-comparison/build_comparison.py
echo "YOLO26s-P2 Phase 1 bundle completed: E2, E3, and conditional E4; each completed experiment has a submission ZIP."
