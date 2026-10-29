#!/bin/bash
# Analysis (goal: Δm 70, 29 Oct): where twins lose AP against their training copies (recognition, scores, boxes)
#SBATCH --job-name=sar-twincopy
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --time=00:45:00
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-tc-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
{ echo "== $(date '+%m-%d %H:%M') twins vs training copies, 340514 (job $SLURM_JOB_ID)"
  python ~/sbatch-staging/twin_vs_copy.py runs/ft40-s1r-fliprefit-340514/train/weights/best.pt \
      results/ft40-s1r-fliprefit-340514/one2many results/twin-copies.json results/twincopy-340514 \
      results/ft40-s1r-twinjit-341749/one2many 2>&1 | grep -v -i -E "warn|_methods|rcount"
  echo TWINCOPY_DONE; } >> ~/sbatch-staging/twin_vs_copy.out
