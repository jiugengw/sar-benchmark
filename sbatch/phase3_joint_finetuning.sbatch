#!/bin/bash
# Prepare all tracked inputs, then run the YOLO26m and RT-DETR-L joint fine-tunes.
# Submit from the checked-out repository with: sbatch sbatch/phase3_joint_finetuning.sbatch
# The job does not submit predictions to the course leaderboard.
#SBATCH --job-name=sar-phase3-joint
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=phase3-joint-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0

if ! git lfs version >/dev/null 2>&1; then
    echo "Git LFS is required to fetch the Phase 3 checkpoints and course data." >&2
    exit 2
fi
git lfs pull --include="data/*.tar.gz,models/yolo/phase3_assets/checkpoints/*.pt"

VENV_DIR=""
if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import torch, torchvision, ultralytics' >/dev/null 2>&1; then
    VENV_DIR=".venv"
else
    if [[ ! -x .phase3-venv/bin/python ]]; then
        python -m venv .phase3-venv
    fi
    if ! .phase3-venv/bin/python -c 'import torch, torchvision, ultralytics' >/dev/null 2>&1; then
        .phase3-venv/bin/python -m pip install --upgrade pip
        .phase3-venv/bin/python -m pip install -r models/yolo/requirements.txt
    fi
    VENV_DIR=".phase3-venv"
fi
source "${VENV_DIR}/bin/activate"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/sar-phase3-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
import torch
import ultralytics

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is visible inside this Slurm allocation")
print(f"PyTorch {torch.__version__}; Ultralytics {ultralytics.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}; visible memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
PY

for checkpoint in \
    models/yolo/phase3_assets/checkpoints/E5_yolo26m_p2_best.pt \
    models/yolo/phase3_assets/checkpoints/E5_yolo26m_image_classifier.pt \
    models/yolo/phase3_assets/checkpoints/E7_rtdetr_l_p2_best.pt \
    models/yolo/phase3_assets/checkpoints/E7_rtdetr_l_image_classifier.pt; do
    if [[ ! -s "$checkpoint" ]]; then
        echo "Missing or empty Phase 3 checkpoint: $checkpoint" >&2
        exit 2
    fi
done

if [[ ! -s data/train.tar.gz || ! -s data/val.tar.gz ]]; then
    echo "Course data archives are missing after Git LFS pull." >&2
    exit 2
fi
mkdir -p data/course-data
if [[ ! -s data/course-data/train/labels.csv || ! -s data/course-data/val/images.json ]]; then
    tar -xzf data/train.tar.gz -C data/course-data
    tar -xzf data/val.tar.gz -C data/course-data
fi

python -u models/yolo/prepare_dataset.py \
    --data data/course-data \
    --split results/yolo26m-100-p2-328339/seed_0/split.json \
    --out data/yolo11s-sar-512 \
    --size 512

JOB_TAG="${SLURM_JOB_ID:-local}"
YOLO_OUTPUT="runs/phase3-yolo26m-joint-${JOB_TAG}"
RTDETR_OUTPUT="runs/phase3-rtdetr-l-joint-${JOB_TAG}"
mkdir -p runs

python -u models/yolo/preflight_rtdetr_joint.py \
    --config models/yolo/experiment_configs/phase3_rtdetr_l.yaml \
    --output "${RTDETR_OUTPUT}/preflight.json"

python -u models/yolo/train_joint.py \
    --config models/yolo/experiment_configs/phase3_yolo26m.yaml \
    --experiment E2 \
    --output "$YOLO_OUTPUT"

python -u models/yolo/train_rtdetr_joint.py \
    --config models/yolo/experiment_configs/phase3_rtdetr_l.yaml \
    --experiment E2 \
    --output "$RTDETR_OUTPUT"

echo "Phase 3 joint fine-tuning finished."
echo "YOLO26m run: $YOLO_OUTPUT"
echo "RT-DETR-L run: $RTDETR_OUTPUT"
echo "Slurm log: phase3-joint-${JOB_TAG}.log"
