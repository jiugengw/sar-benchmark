#!/bin/bash
# PRIVATE diagnostic, never submitted, not in the repo: the M4-oversampling third stage (control 333334) with N extra
# M4-SAR images read from the zip, at the control's compute budget (~497k images seen), scored on our holdout.
#   sbatch m4_curve.sbatch N EPOCHS [control]
#SBATCH --job-name=sar-m4x
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/common/scratch/users/j/jgwong.2023/private-val/%x-%j.out
set -euo pipefail
cd /common/scratch/users/j/jgwong.2023/sar-benchmark
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/sar-m4x-${SLURM_JOB_ID}"
export PYTHONPATH="$PWD:$PWD/models/yolo"
N="$1"; EPOCHS="$2"
if [[ "${3:-}" == control ]]; then  # the control third stage was trained with --skip-evaluation: score it the same way
    python -u -c "
from pathlib import Path
from models.yolo.deadline_recovery import evaluate_export
evaluate_export(Path('runs/ft40-m4repeat-third-333334/train/weights/best.pt'), Path('runs/chain-dota-330938-package/image_classifier.pt'), 'one2many', Path('results/private-m4x0-control-third-333334/one2many'))"
fi
python -u ../private-val/m4_finetune.py --extra "$N" --tag "private-m4x${N}-third-${SLURM_JOB_ID}" \
    --source runs/chain-dota-330938-speckle/train/weights/best.pt \
    --classifier runs/chain-dota-330938-package/image_classifier.pt --seed 0 --epochs "$EPOCHS" --lr0 0.001 \
    --repeat-classes airport,harbor,oil_tank,playground,wind_turbine --repeat-factor 2 --repeat-width 512 --save-period 0
echo "M4X_COMPLETE $N"
