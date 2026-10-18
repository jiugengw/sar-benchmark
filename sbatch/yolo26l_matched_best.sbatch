#!/bin/bash
# Mirror the best local medium recipe on large: 100 baseline + 40 plain + 40 speckle,
# then refit the same classifier and cross-fit box calibration. Existing matched baselines are reused.
# Submit before retrying Dome-DETR: sbatch sbatch/yolo26l_matched_best.sbatch
#SBATCH --job-name=sar-y26l-matched
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="/tmp/sar-y26l-matched-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

python -u models/yolo/matched_large.py --tag "yolo26l-matched-best-${SLURM_JOB_ID}"
