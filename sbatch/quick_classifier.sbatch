#!/bin/bash
#SBATCH --job-name=sar-quick-head
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=runs/sar-quick-head-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/sar-quick-head-${SLURM_JOB_ID}"
python -u yolo_baseline/quick_classifier.py --output "runs/yolo26m_p2_quick_head_${SLURM_JOB_ID}" --max-minutes 21
