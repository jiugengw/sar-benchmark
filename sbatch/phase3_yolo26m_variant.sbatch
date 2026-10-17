#!/bin/bash
# One YOLO26m-P2 Phase 3 joint fine-tune from E5, then holdout scoring on the full and clean subsets.
# Run `git pull` and `git lfs pull` first, then submit from the project directory, for example:
#   PHASE3_CONFIG=yolo_baseline/experiment_configs/phase3_yolo26m_seed1.yaml sbatch sbatch/phase3_yolo26m_variant.sbatch
# Several variants can run at once: shared data is prepared once, under a lock, and never rewritten.
# The job does not submit predictions to the course leaderboard.
#SBATCH --job-name=sar-phase3-variant
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0

CONFIG="${PHASE3_CONFIG:?Set PHASE3_CONFIG to a yolo_baseline/experiment_configs/phase3_yolo26m*.yaml file}"
LABEL="$(basename "$CONFIG" .yaml)"
DATA="data/course-data"
DATASET="data/yolo11s-sar-512"
SPLIT="results/yolo26m-100-p2-328339/seed_0/split.json"
E5_WEIGHTS="yolo_baseline/phase3_assets/checkpoints/E5_yolo26m_p2_best.pt"
RUN_ROOT="runs/${LABEL}-${SLURM_JOB_ID}"
RESULT_ROOT="results/${LABEL}-${SLURM_JOB_ID}/seed_0"

for required in "$CONFIG" sarbench/train_twins.json; do
    if [[ ! -s "$required" ]]; then
        echo "Missing $required (run git pull)" >&2
        exit 2
    fi
done
# Git LFS pointer files are about 130 bytes; the smallest real asset here is about 70 kB.
for asset in "$E5_WEIGHTS" yolo_baseline/phase3_assets/checkpoints/E5_yolo26m_image_classifier.pt \
        data/train.tar.gz data/val.tar.gz; do
    if [[ ! -s "$asset" || $(stat -c %s "$asset") -lt 4096 ]]; then
        echo "Missing or not yet fetched: $asset (run git lfs pull)" >&2
        exit 2
    fi
done

VENV_DIR=""
for candidate in .venv .phase3-venv; do
    if [[ -x "$candidate/bin/python" ]] && "$candidate/bin/python" -c 'import torch, torchvision, ultralytics' >/dev/null 2>&1; then
        VENV_DIR="$candidate"
        break
    fi
done
if [[ -z "$VENV_DIR" ]]; then
    echo "No environment with torch, torchvision and ultralytics; run sbatch/phase3_joint_finetuning.sbatch once to create .phase3-venv." >&2
    exit 2
fi
source "${VENV_DIR}/bin/activate"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/sar-phase3-variant-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/yolo_baseline${PYTHONPATH:+:${PYTHONPATH}}"

# Extract the course data and convert the saved split only if a previous job has not already done so.
(
    flock 9
    if [[ ! -s "$DATA/train/labels.csv" || ! -s "$DATA/val/images.json" ]]; then
        mkdir -p "$DATA"
        tar -xzf data/train.tar.gz -C "$DATA"
        tar -xzf data/val.tar.gz -C "$DATA"
    fi
    if [[ ! -s "$DATASET/dataset.yaml" ]]; then
        python -u yolo_baseline/prepare_dataset.py --data "$DATA" --split "$SPLIT" --out "$DATASET" --size 512
    fi
) 9>data/.prepare.lock

mkdir -p runs "$RESULT_ROOT"
python -u yolo_baseline/train_joint.py --config "$CONFIG" --experiment E2 --output "$RUN_ROOT"

python -u yolo_baseline/run_experiment.py \
    --data "$DATA" --dataset-yaml "$DATASET/dataset.yaml" --dataset-root "$DATASET" --split "$SPLIT" \
    --baseline-results results/tm-48baseline-326894/terramind_pretrained_moelora/seed_0 \
    --baseline-checkpoint checkpoints/tm-48baseline-326894/terramind_pretrained_moelora/seed_0/model.pt \
    --weights "$E5_WEIGHTS" --train-output "$RUN_ROOT" --results-output "$RESULT_ROOT" \
    --epochs 15 --batch 16 --workers 4 \
    --classifier-checkpoint "$RUN_ROOT/image_classifier.pt" \
    --model-label "YOLO26m-P2 joint (${LABEL})" --eval-only

python -m sarbench.twins score "$RESULT_ROOT" || true  # summary only; metrics.json already holds both
echo "Run: $RUN_ROOT"
echo "Scores (full and clean holdout): $RESULT_ROOT/metrics.json"
