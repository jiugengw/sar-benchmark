#!/bin/bash
#SBATCH --job-name=sar-tm-def-unc
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/tm-default-uncertainty-%j.out

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

RUN_ROOT="runs/tm-default-uncertainty-${SLURM_JOB_ID}"
RESULT_ROOT="results/tm-default-uncertainty-${SLURM_JOB_ID}"
CHECKPOINT_ROOT="checkpoints/tm-default-uncertainty-${SLURM_JOB_ID}"
SUBMISSION_ROOT="nonsubmissions/151026/tests-no-submit/tm-default-uncertainty-${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$SUBMISSION_ROOT"

python -u train.py --data data/course-data --backbone terramind --init pretrained --adapt moelora \
    --epochs 48 --batch-size 16 --lr 1e-4 --seed 0 --workers 4 --eval-every 2 \
    --box-loss-weight 2 --photometric --select-best --anchor-profile default \
    --loss-weighting uncertainty --out "$RUN_ROOT" \
    --results "$RESULT_ROOT" --checkpoints "$CHECKPOINT_ROOT"
python -u calibrate_submission.py --run "$RUN_ROOT/terramind_pretrained_moelora_uncertainty" \
    --out "$SUBMISSION_ROOT/terramind_48ep_default_anchors_uncertainty_calibrated.zip"
