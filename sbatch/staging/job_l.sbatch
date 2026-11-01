#!/bin/bash
# job_l: where the holdout loses bridges and playgrounds (job_l.py); rescoring and crop sheets only, no inference.
#   sbatch --job-name=job3 --constraint=<free type> ~/sbatch-staging/job_l.sbatch
#SBATCH --job-name=job3
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
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-l-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 SAR_INPUT_FILTER=none
OUT=~/sbatch-staging/job_l.out
echo "== $(date '+%m-%d %H:%M') job $SLURM_JOB_ID on $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $OUT
python -u ~/sbatch-staging/job_l.py results/misses-340514 2>&1 | grep -v -i warn >> $OUT
echo "JOB_L_END $(date '+%m-%d %H:%M')" >> $OUT
