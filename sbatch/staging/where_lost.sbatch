#!/bin/bash
# Goal Δm 70 (user, 29 Oct: "continue trying, think of why we not good, twins performance"): where the test mAP is lost
# by holdout group and class, and what is left in the twin gap (labels vs boxes, sharpness). CPU only, analysis only.
#SBATCH --job-name=sar-where-lost
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --prefer=l40s
#SBATCH --time=02:00:00
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONHASHSEED=0 MPLCONFIGDIR=/tmp/sar-wl-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
{ echo "== $(date '+%m-%d %H:%M') where the test mAP is lost; the twin gap (job $SLURM_JOB_ID)"
  python -u ~/sbatch-staging/where_lost.py $CP 2>&1 | grep -v -i warn; } >> ~/sbatch-staging/where_lost.out
