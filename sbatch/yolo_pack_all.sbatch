#!/bin/bash
#SBATCH --job-name=sar-yolo-pack
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
export MPLCONFIGDIR="/tmp/yolo-pack-${SLURM_JOB_ID}-matplotlib"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u models/yolo/make_all_submissions.py
python reports/training-comparison/build_comparison.py
