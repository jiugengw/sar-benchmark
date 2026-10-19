#!/bin/bash
# Diagnostic: the best model on the holdout fed at larger input sizes (models/yolo/resolution_probe.py); never a submission.
#   sbatch sbatch/resolution_probe.sbatch runs/ft40-memorise-lr2-332326/train/weights/best.pt
#SBATCH --job-name=sar-resprobe
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/sar-resprobe-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

python -u models/yolo/resolution_probe.py --weights "${1:?detector weights}" ${2:+--sizes "$2"}
