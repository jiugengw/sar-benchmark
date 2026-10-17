#!/bin/bash
#SBATCH --job-name=sar-y26-recovery
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:30:00
#SBATCH --output=runs/sar-y26-recovery-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/sar-y26-recovery-${SLURM_JOB_ID}"
python -u yolo_baseline/deadline_recovery.py \
    --output "runs/deadline-yolo26m-${SLURM_JOB_ID}" --epochs 40 --stages 2
