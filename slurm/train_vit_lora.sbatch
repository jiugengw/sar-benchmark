#!/usr/bin/env bash
#SBATCH --job-name=sar-vit-lora
#SBATCH --partition=student
#SBATCH --account=student
#SBATCH --qos=studentqos
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

DATA_DIR="${DATA_DIR:-data/course-data}"
if [[ ! -f "$DATA_DIR/train/labels.csv" || ! -f "$DATA_DIR/train/instances.json" ]]; then
    echo "Course data not found under $DATA_DIR. See slurm/README.md for setup." >&2
    exit 1
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this job; check the cluster's GPU setup.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

python train.py \
    --data "$DATA_DIR" \
    --backbone vit \
    --init pretrained \
    --adapt lora \
    --seed 0 \
    --eval-every 1
