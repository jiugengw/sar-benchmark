#!/bin/bash
# Dome-DETR-M (AI-TOD optical weights, 24.4M parameters) fine-tuned on our SAR split at 512 px, then the holdout
# scored full / clean against the YOLO control (continue_2: mAP 44.98 full / 41.89 clean).
# One-time setup on the login node: bash scripts/setup_dome.sh
# Submit from the project root: sbatch sbatch/dome_detr.sbatch            (EPOCHS=48 BATCH=8 by default)
#SBATCH --job-name=sar-dome
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
source .dome-venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="/tmp/sar-dome-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

TAG="dome-m-${SLURM_JOB_ID}"
python -u models/dome/train.py --tag "$TAG" --epochs "${EPOCHS:-48}" --batch "${BATCH:-8}" "$@"
python -u models/dome/export_predictions.py --tag "$TAG"
echo "Dome-DETR finished: runs/$TAG, results/$TAG"
