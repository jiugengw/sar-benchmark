#!/bin/bash
#SBATCH --job-name=sar-small48
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/small48-%j.out

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

EXPERIMENT="${1:?Pass vit, resnet34 or terramind_uncertainty}"
BACKBONE="$EXPERIMENT"
LOSS_WEIGHTING=fixed
EXTRA_ARGS=()
case "$EXPERIMENT" in
    vit)
        ADAPT=moelora
        BACKBONE_LR=1e-4
        WEIGHTS=checkpoints/vit_pretrained_base_seed0.pt
        ;;
    resnet34)
        ADAPT=full
        BACKBONE_LR=2e-5
        WEIGHTS=checkpoints/resnet34-imagenet1k-b627a593.pth
        ;;
    terramind_uncertainty)
        BACKBONE=terramind
        ADAPT=moelora
        BACKBONE_LR=1e-4
        LOSS_WEIGHTING=uncertainty
        WEIGHTS=
        ;;
    *) echo "Expected vit, resnet34 or terramind_uncertainty" >&2; exit 2 ;;
esac
if [[ -n "$WEIGHTS" ]]; then
    test -s "$WEIGHTS"
    EXTRA_ARGS+=(--backbone-weights "$WEIGHTS")
fi
CONFIGURATION="${BACKBONE}_pretrained_${ADAPT}"
if [[ "$LOSS_WEIGHTING" != fixed ]]; then
    CONFIGURATION="${CONFIGURATION}_${LOSS_WEIGHTING}"
fi
RUN_ROOT="runs/small48-${BACKBONE}-${SLURM_JOB_ID}"
RESULT_ROOT="results/small48-${BACKBONE}-${SLURM_JOB_ID}"
CHECKPOINT_ROOT="checkpoints/small48-${BACKBONE}-${SLURM_JOB_ID}"
SUBMISSION_ROOT="submissions/151026/tests-no-submit/small48-${BACKBONE}-${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$SUBMISSION_ROOT"

python -u train.py --data data/course-data --backbone "$BACKBONE" --init pretrained --adapt "$ADAPT" \
    --backbone-lr "$BACKBONE_LR" --loss-weighting "$LOSS_WEIGHTING" \
    --epochs 48 --batch-size 16 --lr 1e-4 --seed 0 --workers 4 --eval-every 2 \
    --box-loss-weight 2 --photometric --select-best --anchor-profile small_extreme \
    --out "$RUN_ROOT" --results "$RESULT_ROOT" --checkpoints "$CHECKPOINT_ROOT" "${EXTRA_ARGS[@]}"
python -u calibrate_submission.py --run "$RUN_ROOT/$CONFIGURATION" \
    --out "$SUBMISSION_ROOT/${BACKBONE}_small_extreme_calibrated.zip"
