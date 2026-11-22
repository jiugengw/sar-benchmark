#!/bin/bash
# TerraMind fusion smoke test (29 Oct): equality at start, a 2% one-epoch
# training through Ultralytics, reload, make_submission, head paths, step time. Results: tm_smoke.out
#SBATCH --job-name=sar-tm-smoke
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-tms-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
OUT=/tmp/sar-tm-smoke-$SLURM_JOB_ID; mkdir -p $OUT
{ echo "== $(date '+%m-%d %H:%M') TerraMind fusion smoke test (job $SLURM_JOB_ID) on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  python -u ~/sbatch-staging/tm_smoke.py $OUT 2>&1 | grep -v -i -E "warn|it/s]|Freezing|requires_grad=True" | tail -60; } >> ~/sbatch-staging/tm_smoke.out
rm -rf $OUT
