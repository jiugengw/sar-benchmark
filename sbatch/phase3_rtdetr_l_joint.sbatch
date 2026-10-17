#!/bin/bash
# Standalone Phase 3 RT-DETR-L joint fine-tune from the completed E7 checkpoints.
# Submit from the repository root with: sbatch sbatch/phase3_rtdetr_l_joint.sbatch
# This is the RT-DETR half of the documented Phase 3 plan.
#SBATCH --job-name=sar-phase3-rtdetr-l
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
export MPLCONFIGDIR="/tmp/sar-phase3-rtdetr-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/yolo_baseline${PYTHONPATH:+:${PYTHONPATH}}"

for input in \
    yolo_baseline/phase3_assets/checkpoints/E7_rtdetr_l_p2_best.pt \
    yolo_baseline/phase3_assets/checkpoints/E7_rtdetr_l_image_classifier.pt \
    yolo_baseline/experiment_configs/phase3_rtdetr_l.yaml \
    data/course-data/train/labels.csv \
    data/yolo11s-sar-512/dataset.yaml \
    results/yolo26m-100-p2-328339/seed_0/split.json; do
    if [[ ! -s "$input" ]]; then
        echo "Missing RT-DETR fine-tune input: $input" >&2
        exit 2
    fi
done

python - <<'PY'
import torch
import ultralytics

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is visible inside this Slurm allocation")
print(f"PyTorch {torch.__version__}; Ultralytics {ultralytics.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}; visible memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
PY

JOB_TAG="${SLURM_JOB_ID}"
OUTPUT="runs/phase3-rtdetr-l-joint-${JOB_TAG}"
mkdir -p "$OUTPUT"

python -u yolo_baseline/preflight_rtdetr_joint.py \
    --config yolo_baseline/experiment_configs/phase3_rtdetr_l.yaml \
    --output "$OUTPUT/preflight.json"

python -u yolo_baseline/train_rtdetr_joint.py \
    --config yolo_baseline/experiment_configs/phase3_rtdetr_l.yaml \
    --experiment E2 \
    --output "$OUTPUT"

echo "RT-DETR-L joint fine-tune finished: $OUTPUT"
