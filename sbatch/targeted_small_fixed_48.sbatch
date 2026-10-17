#!/bin/bash
#SBATCH --job-name=sar-small-fixed48
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/small-fixed-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/sarbench-matplotlib
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN_ROOT="runs/small-fixed-${SLURM_JOB_ID}"
RESULT_ROOT="results/small-fixed-${SLURM_JOB_ID}"
CHECKPOINT_ROOT="checkpoints/small-fixed-${SLURM_JOB_ID}"
SUBMISSION_ROOT="nonsubmissions/151026/small-fixed-${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$SUBMISSION_ROOT"

python -u models/terramind/train.py --data data/course-data --backbone terramind --init pretrained --adapt moelora \
    --epochs 48 --batch-size 16 --lr 1e-4 --seed 0 --workers 4 --eval-every 2 \
    --box-loss-weight 2 --photometric --select-best --anchor-profile small_extreme \
    --loss-weighting fixed --out "$RUN_ROOT" --results "$RESULT_ROOT" \
    --checkpoints "$CHECKPOINT_ROOT"
python -u models/terramind/calibrate_submission.py --run "$RUN_ROOT/terramind_pretrained_moelora" \
    --out "$SUBMISSION_ROOT/terramind_48ep_small_anchors_fixed_calibrated.zip"
python reports/training-comparison/build_comparison.py || true
