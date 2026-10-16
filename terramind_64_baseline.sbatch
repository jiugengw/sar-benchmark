#!/bin/bash
#SBATCH --job-name=sar-tm-64base
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/tm-64baseline-%j.out

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

RUN_ROOT="runs/tm-64baseline-${SLURM_JOB_ID}"
RESULT_ROOT="results/tm-64baseline-${SLURM_JOB_ID}"
CHECKPOINT_ROOT="checkpoints/tm-64baseline-${SLURM_JOB_ID}"
SUBMISSION_ROOT="submissions/151026/tests-no-submit/tm-64baseline-${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$SUBMISSION_ROOT"

python -u train.py --data data/course-data --backbone terramind --init pretrained --adapt moelora \
    --epochs 64 --batch-size 16 --lr 1e-4 --seed 0 --workers 4 --eval-every 2 \
    --box-loss-weight 1 --anchor-profile default --loss-weighting fixed --select-best \
    --out "$RUN_ROOT" --results "$RESULT_ROOT" --checkpoints "$CHECKPOINT_ROOT"
python -u calibrate_submission.py --run "$RUN_ROOT/terramind_pretrained_moelora" \
    --out "$SUBMISSION_ROOT/terramind_64epoch_baseline_calibrated.zip"
