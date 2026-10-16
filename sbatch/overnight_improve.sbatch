#!/bin/bash
#SBATCH --job-name=sar-improve
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=07:00:00
#SBATCH --array=0-3%2
#SBATCH --output=runs/overnight-improve-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
TASK_ROOT="runs/overnight-improve-${SLURM_ARRAY_JOB_ID}"
mkdir -p "$TASK_ROOT" "submissions/overnight-improve-${SLURM_ARRAY_JOB_ID}"
SUBMISSION_ROOT="submissions/overnight-improve-${SLURM_ARRAY_JOB_ID}"

if [[ "$SLURM_ARRAY_TASK_ID" == 3 ]]; then
    # Existing checkpoints: no new training, select calibration using local labels only.
    for MODEL_NAME in terramind_pretrained_moelora terramind_pretrained_moelora_box_localization_seed0; do
        if [[ "$MODEL_NAME" == terramind_pretrained_moelora ]]; then
            MODEL_DIR="checkpoints/$MODEL_NAME/seed_0"
        else
            MODEL_DIR="checkpoints/$MODEL_NAME"
        fi
        python -u calibrate_submission.py --run "$MODEL_DIR" \
            --holdout "results/$MODEL_NAME/instances_val_holdout.json" \
            --out "$SUBMISSION_ROOT/${MODEL_NAME}_calibrated.zip"
    done
    exit 0
fi

EXTRA_ARGS=()
case "$SLURM_ARRAY_TASK_ID" in
    0) TRIAL=longer_baseline; BOX_WEIGHT=1 ;;
    1) TRIAL=longer_boxloss; BOX_WEIGHT=2 ;;
    2) TRIAL=longer_boxloss_photometric; BOX_WEIGHT=2; EXTRA_ARGS+=(--photometric) ;;
    *) exit 2 ;;
esac
python -u train.py --data data/course-data --backbone terramind --init pretrained --adapt moelora \
    --epochs 36 --batch-size 16 --lr 1e-4 --seed 0 --workers 4 --eval-every 2 \
    --box-loss-weight "$BOX_WEIGHT" --select-best \
    --out "$TASK_ROOT/$TRIAL" --results "results/overnight-improve-${SLURM_ARRAY_JOB_ID}/$TRIAL" \
    --checkpoints "checkpoints/overnight-improve-${SLURM_ARRAY_JOB_ID}/$TRIAL" "${EXTRA_ARGS[@]}"
python -u calibrate_submission.py --run "$TASK_ROOT/$TRIAL/terramind_pretrained_moelora" \
    --out "$SUBMISSION_ROOT/${TRIAL}_calibrated.zip"
