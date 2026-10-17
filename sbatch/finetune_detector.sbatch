#!/bin/bash
# 40 more epochs on the current best detector (continue_1) with ONE change, compared against continue_2.
# Submit from the project directory, giving a short name and the change, e.g.:
#   sbatch sbatch/finetune_detector.sbatch nwd --nwd-weight 0.5
#   sbatch sbatch/finetune_detector.sbatch speckle --speckle-prob 0.5
#SBATCH --job-name=sar-ft40
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
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="/tmp/sar-ft40-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

NAME="${1:?Give a short experiment name, e.g. nwd}"
shift
python -u models/yolo/finetune_detector.py --tag "ft40-${NAME}-${SLURM_JOB_ID}" "$@"
