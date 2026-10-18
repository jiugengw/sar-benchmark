#!/bin/bash
# Experiment A: bridge context boxes. Fine-tune the DOTA m chain detector for 40 epochs on labels where every bridge
# box is 3x larger (river and road around it), then shrink predicted bridge boxes back by 3x and score.
# Compare with the same 40-epoch fine-tune on normal labels (the control):
#   sbatch sbatch/context_boxes.sbatch                 # context boxes
#   sbatch sbatch/context_boxes.sbatch control         # identical fine-tune, normal labels
# Optional environment: CLASSES (default bridge), FACTOR (default 3).
#SBATCH --job-name=sar-ctx
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="/tmp/sar-ctx-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

MODE="${1:-context}"
CLASSES="${CLASSES:-bridge}"
FACTOR="${FACTOR:-3}"
SOURCE=runs/chain-dota-330938-speckle/train/weights/best.pt
CLASSIFIER=runs/chain-dota-330938-package/image_classifier.pt

if [[ "$MODE" == control ]]; then
    TAG="ft40-ctxcontrol-${SLURM_JOB_ID}"
    python -u models/yolo/finetune_detector.py --tag "$TAG" --source "$SOURCE" --classifier "$CLASSIFIER"
    echo "CONTROL_COMPLETE results/$TAG/one2many"
    exit 0
fi

DATA="data/yolo-sar-512-ctx${FACTOR}-${CLASSES//,/-}"
if [[ ! -f "$DATA/dataset.yaml" ]]; then
    python -u models/yolo/build_variant_dataset.py --out "$DATA" --context-classes "$CLASSES" --context-factor "$FACTOR"
fi
TAG="ft40-ctx${FACTOR}-${CLASSES//,/-}-${SLURM_JOB_ID}"
python -u models/yolo/finetune_detector.py --tag "$TAG" --source "$SOURCE" --classifier "$CLASSIFIER" \
    --data "$DATA/dataset.yaml"
# The fine-tune's own scores above use unshrunk boxes; the real result is after shrinking.
for BRANCH in one2many one2one; do
    python -u models/yolo/context_boxes.py "results/$TAG/$BRANCH" --classes "$CLASSES" --factor "$FACTOR"
done
echo "CONTEXT_COMPLETE results/$TAG/one2many-shrunk (compare with the control and with chain-dota-330938: bridge 6.2)"
