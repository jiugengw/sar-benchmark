#!/bin/bash
# job_s: rescore existing holdout predictions step by step and by image group (job_s.py); no training, no inference.
#   sbatch --job-name=job-s1 ~/sbatch-staging/job_s.sbatch
#SBATCH --job-name=job-s
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-s-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 SAR_INPUT_FILTER=none
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u ~/sbatch-staging/job_s.py
echo "end $(date '+%m-%d %H:%M')"
