#!/bin/bash
# job_k: two recall checks on the holdout for the Phase 1 race (1 Nov; job_k.py): B. one2one + one2many boxes merged,
# A. image-class rescoring. Inference for B only (one2one boxes of 340514, ~5 min), the rest rescoring.
#   sbatch --job-name=job2 --constraint=<free type> ~/sbatch-staging/job_k.sbatch
#SBATCH --job-name=job2
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-k-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 SAR_INPUT_FILTER=none
OUT=~/sbatch-staging/job_k.out
echo "== $(date '+%m-%d %H:%M') job $SLURM_JOB_ID on $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $OUT
python -u ~/sbatch-staging/job_k.py export && python -u ~/sbatch-staging/job_k.py merge 2>&1 | grep -v -i warn >> $OUT
python -u ~/sbatch-staging/job_k.py rescore 2>&1 | grep -v -i warn >> $OUT
echo "JOB_K_END $(date '+%m-%d %H:%M')" >> $OUT
