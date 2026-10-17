#!/bin/bash
#SBATCH --job-name=sar-y26-screen
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=runs/sar-y26-screen-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python -u yolo_baseline/screen_detection_branch.py \
    --weights yolo_baseline/phase3_assets/checkpoints/E5_yolo26m_p2_best.pt \
    --output "results/deadline-screen-${SLURM_JOB_ID}/branch_metrics.json"
