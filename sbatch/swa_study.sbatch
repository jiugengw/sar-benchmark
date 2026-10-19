#!/bin/bash
# Weight averaging of a run's last checkpoints vs its best single checkpoint, on the holdout (models/yolo/swa_study.py).
#   sbatch sbatch/swa_study.sbatch runs/ft40-memorise-lr2-332326 runs/memorise-lr2-332326-package/image_classifier.pt swa-memorise-lr2
#SBATCH --job-name=sar-swa
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
export MPLCONFIGDIR="/tmp/sar-swa-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

python -u models/yolo/swa_study.py --run "${1:?run folder}" --classifier "${2:?classifier}" --tag "${3:?tag}"
