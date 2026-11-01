#!/bin/bash
# job_l_boost: would raising low scores help bridges and playgrounds (job_l_boost.py)? Rescoring only.
#   sbatch --job-name=job4 --constraint=<free type> ~/sbatch-staging/job_l_boost.sbatch
#SBATCH --job-name=job4
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-lb-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 SAR_INPUT_FILTER=none
OUT=~/sbatch-staging/job_l_boost.out
echo "== $(date '+%m-%d %H:%M') job $SLURM_JOB_ID on $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $OUT
python -u ~/sbatch-staging/job_l_boost.py 2>&1 | grep -v -i warn >> $OUT
echo "JOB_LB_END $(date '+%m-%d %H:%M')" >> $OUT
