#!/bin/bash
# Train and package three 100-epoch P2 detectors in one sequential allocation.
# Submit from the repository root with sbatch sbatch/sar_p2_three_model_comparison.sbatch.
# The student partition permits up to 24 hours per job.
#SBATCH --job-name=sar-p2-three-models
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
export PACKAGE_ROOT="nonsubmissions/$(date +%d%m%y)"  # unsubmitted packages go under the job start date
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/sar-p2-three-models-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

DATA="data/course-data"
DATASET="data/yolo11s-sar-512"
SPLIT="runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json"
BASELINE_RESULTS="results/tm-48baseline-326894/terramind_pretrained_moelora/seed_0"
BASELINE_CHECKPOINT="checkpoints/tm-48baseline-326894/terramind_pretrained_moelora/seed_0/model.pt"
MANIFEST="$DATASET/conversion_manifest.json"

run_and_package() {
    local model_key="$1"
    local family="$2"
    local scale="$3"
    local label="$4"
    local run_root="runs/${model_key}-100-p2-${SLURM_JOB_ID}"
    local result_root="results/${model_key}-100-p2-${SLURM_JOB_ID}/seed_0"
    local package_dir="${PACKAGE_ROOT}/${model_key}-100-p2-${SLURM_JOB_ID}"
    local weights="pretrained/${model_key}.pt"

    if [[ ! -f "$weights" ]]; then
        echo "Missing pretrained checkpoint: $weights" >&2
        exit 1
    fi
    mkdir -p "$run_root" "$result_root" "$package_dir/scoring_result"
    touch "$package_dir/scoring_result/.gitkeep"

    if [[ "$family" == "yolo26" ]]; then
        python -u models/yolo/train_yolo26s_p2.py \
            --scale "$scale" --weights "$weights" --train-output "$run_root" \
            --epochs 100 --batch 16 --workers 4
    else
        python -u models/yolo/train_100_epoch.py \
            --scale "$scale" --variant p2 --weights "$weights" \
            --dataset-yaml "$DATASET/dataset.yaml" --classes "$MANIFEST" \
            --train-output "$run_root" --epochs 100 --batch 16 --workers 4
    fi

    python -u models/yolo/train_image_head.py \
        --detector-checkpoint "$run_root/train/weights/best.pt" \
        --data "$DATA" --dataset-root "$DATASET" --split "$SPLIT" \
        --output "$run_root/image_classifier.pt" --epochs 100 --batch 32 --workers 4

    python -u models/yolo/run_experiment.py \
        --data "$DATA" --dataset-yaml "$DATASET/dataset.yaml" \
        --dataset-root "$DATASET" --split "$SPLIT" \
        --baseline-results "$BASELINE_RESULTS" --baseline-checkpoint "$BASELINE_CHECKPOINT" \
        --weights "$weights" --train-output "$run_root" --results-output "$result_root" \
        --epochs 100 --batch 16 --workers 4 \
        --classifier-checkpoint "$run_root/image_classifier.pt" \
        --model-label "$label" --eval-only

    python -u models/yolo/make_submission.py \
        --weights "$run_root/train/weights/best.pt" \
        --classifier-checkpoint "$run_root/image_classifier.pt" \
        --box-class-policy native_detector_classes --data "$DATA" --classes "$MANIFEST" \
        --out "$package_dir/${model_key}_100_p2_${SLURM_JOB_ID}.zip" --batch 16 --workers 4

    cp "$result_root/metrics.json" "$package_dir/metrics.json"
    cp "$run_root/compute.json" "$package_dir/compute.json"
    cp "$run_root/image_classifier.metrics.json" "$package_dir/classifier.metrics.json"
    if [[ -f "$run_root/benchmark_settings.json" ]]; then
        cp "$run_root/benchmark_settings.json" "$package_dir/config.json"
    fi
    if [[ -f "$run_root/experiment_config.json" ]]; then
        cp "$run_root/experiment_config.json" "$package_dir/config.json"
    fi
}

run_and_package yolo26l yolo26 l "YOLO26L-100-P2"
run_and_package yolo11m yolo11 m "YOLO11M-100-P2"
run_and_package yolo11l yolo11 l "YOLO11L-100-P2"

echo "Completed and packaged YOLO26L-P2, YOLO11M-P2, and YOLO11L-P2."
