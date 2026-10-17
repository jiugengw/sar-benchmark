#!/bin/bash
# Resume the saved DenoDet V2 optimizer/model state through epoch 48 and export.
#SBATCH --job-name=sar-denodet-resume
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0 CUDA/12.8.0
source .denodet-venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export CUDA_HOME=/opt/apps/software/CUDA/12.8.0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/third_party/GrokSAR${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/sar-denodet-resume-${SLURM_JOB_ID}"
python -u models/denodet/train.py \
    --output runs/denodet-v2-329477 --resume --epochs 48 \
    --batch 8 --workers 4
