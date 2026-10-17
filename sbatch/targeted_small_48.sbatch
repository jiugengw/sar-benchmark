#!/bin/bash
#SBATCH --job-name=sar-targeted48
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=runs/targeted-small-%A_%a.out

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

COMMON=(--data data/course-data --backbone terramind --init pretrained --adapt moelora
    --epochs 48 --batch-size 16 --lr 1e-4 --seed 0 --workers 4 --eval-every 2
    --box-loss-weight 2 --photometric --select-best --anchor-profile small_extreme
    --loss-weighting uncertainty)
RUN_ROOT="runs/targeted-small-${SLURM_ARRAY_JOB_ID}"
RESULT_ROOT="results/targeted-small-${SLURM_ARRAY_JOB_ID}"
CHECKPOINT_ROOT="checkpoints/targeted-small-${SLURM_ARRAY_JOB_ID}"

case "$SLURM_ARRAY_TASK_ID" in
    0)
        echo "Experiment: 15.32 recipe + tiny-object image sampling"
        python -u models/terramind/train.py "${COMMON[@]}" --small-object-sampling \
            --out "$RUN_ROOT/small_sampling" --results "$RESULT_ROOT/small_sampling" \
            --checkpoints "$CHECKPOINT_ROOT/small_sampling"
        ;;
    1)
        echo "Experiment: 15.32 recipe + stride-4 image detail branch"
        python -u models/terramind/train.py "${COMMON[@]}" --detail-branch \
            --out "$RUN_ROOT/detail_branch" --results "$RESULT_ROOT/detail_branch" \
            --checkpoints "$CHECKPOINT_ROOT/detail_branch"
        ;;
    *) exit 2 ;;
esac
