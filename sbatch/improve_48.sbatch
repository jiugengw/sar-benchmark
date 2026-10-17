#!/bin/bash
#SBATCH --job-name=sar-48epoch
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-1%2
#SBATCH --output=runs/improve-48-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

case "$SLURM_ARRAY_TASK_ID" in
    0) TRIAL=boxloss_photometric; ANCHORS=default ;;
    1) TRIAL=boxloss_photometric_large_anchors; ANCHORS=large ;;
    *) exit 2 ;;
esac

RUN_ROOT="runs/improve-48-${SLURM_ARRAY_JOB_ID}"
SUBMISSION_ROOT="nonsubmissions/improve-48-${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RUN_ROOT" "$SUBMISSION_ROOT"
echo "Experiment: $TRIAL; 48 epochs; anchors=$ANCHORS"
python -u models/terramind/train.py --data data/course-data --backbone terramind --init pretrained --adapt moelora \
    --epochs 48 --batch-size 16 --lr 1e-4 --seed 0 --workers 4 --eval-every 2 \
    --box-loss-weight 2 --photometric --select-best --anchor-profile "$ANCHORS" \
    --out "$RUN_ROOT/$TRIAL" --results "results/improve-48-${SLURM_ARRAY_JOB_ID}/$TRIAL" \
    --checkpoints "checkpoints/improve-48-${SLURM_ARRAY_JOB_ID}/$TRIAL"
python -u models/terramind/calibrate_submission.py --run "$RUN_ROOT/$TRIAL/terramind_pretrained_moelora" \
    --out "$SUBMISSION_ROOT/${TRIAL}_calibrated.zip"
