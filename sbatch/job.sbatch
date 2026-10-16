#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

# Check student partition node states and GPU types before submitting:
# sinfo -p student -N -o '%N %t %G'

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"

# Activate your Python environment before running sbatch, or add its activation here.
# Example (venv): source .venv/bin/activate
# Example (conda): source /path/to/miniconda3/etc/profile.d/conda.sh
#                 conda activate your-environment

# EDIT THESE SETTINGS for each experiment.
BACKBONE=vit          # vit or terramind
INIT=pretrained       # pretrained or scratch (scratch requires ADAPT=full)
ADAPT=lora            # lora, moelora, or full
EPOCHS=24
BATCH_SIZE=16
LR=1e-4
SEED=0                # also controls the local train/validation split
UNFREEZE_LAST=0       # 0 = adapters only; pretrained lora/moelora supports 1..12
UNFROZEN_LR=2e-5
EVAL_EVERY=4          # use 1 to validate every epoch
WORKERS=4
LIMIT=                # blank = all training data; use 256 with EPOCHS=1 for a smoke test

# Example experiments: change BACKBONE and ADAPT above.
# ViT + LoRA:           BACKBONE=vit,       ADAPT=lora
# ViT + MoE-LoRA:       BACKBONE=vit,       ADAPT=moelora
# TerraMind + LoRA:     BACKBONE=terramind, ADAPT=lora
# TerraMind + MoE-LoRA: BACKBONE=terramind, ADAPT=moelora
# Full fine-tuning exceeds the documented course trainable-parameter budget.

# Each submission gets separate output folders to preserve previous experiments.
RUN_ID="${BACKBONE}_${INIT}_${ADAPT}_epochs${EPOCHS}_lr${LR}_seed${SEED}_job${SLURM_JOB_ID}"
EXTRA_ARGS=()
if [[ -n "$LIMIT" ]]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

echo "Experiment: $RUN_ID"
python train.py \
    --data "${DATA_DIR:-data/course-data}" \
    --backbone "$BACKBONE" \
    --init "$INIT" \
    --adapt "$ADAPT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed "$SEED" \
    --unfreeze-last "$UNFREEZE_LAST" \
    --unfrozen-lr "$UNFROZEN_LR" \
    --eval-every "$EVAL_EVERY" \
    --workers "$WORKERS" \
    --out "runs/$RUN_ID" \
    --results "results/$RUN_ID" \
    --checkpoints "checkpoints/$RUN_ID" \
    "${EXTRA_ARGS[@]}"
