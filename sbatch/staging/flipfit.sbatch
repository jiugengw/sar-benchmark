#!/bin/bash
# Analysis (goal: Δm 70, 28 Oct): is the flip-memorised holdout refit under-fit on flipped views of its own training images?
#SBATCH --job-name=sar-flipfit
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --time=00:45:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-ff-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
{ echo "== $(date '+%m-%d %H:%M') flip fit of 340514 on its own training images (job $SLURM_JOB_ID)"
  python ~/sbatch-staging/flipfit.py runs/ft40-s1r-fliprefit-340514/train/weights/best.pt results/flipfit-340514 2>&1 | grep -v -i -E "warn|_methods|rcount"
  echo FLIPFIT_DONE; } >> ~/sbatch-staging/flipfit.out
