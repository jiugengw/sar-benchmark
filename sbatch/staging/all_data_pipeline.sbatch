#!/bin/bash
# The best recipe so far (memorise lr 0.002, official Δm +49.21) retrained on ALL labelled images: the 8,453 training
# images plus the 939 holdout images. Same stages and settings as the control:
#   100 epochs from DOTA weights -> +40 plain (lr 0.002) -> +40 speckle (lr 0.001) -> +40 third stage (lr 0.001)
#   -> memorise: 40 epochs without augmentation (lr 0.002) -> 1024-unit image classifier, also on all images
#   (240 epochs, last kept) -> official-validation ZIP.
# Every stage keeps its LAST epoch (the holdout is in the training data, so it cannot pick epochs), and the box nudge
# is the control's decision (rejected). Holdout scores printed by this job are optimistic: judge it by Codabench.
#   sbatch sbatch/all_data_pipeline.sbatch
#SBATCH --job-name=sar-alldata
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
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
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="/tmp/sar-alldata-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

WEIGHTS=pretrained/yolo26m-obb.pt
echo "23e0630f66857cf4b87535f6e705b065f1e8a33603640b8e61ace85b75312903  $WEIGHTS" | sha256sum --check
TAG="alldata-${SLURM_JOB_ID}"
BASE="$(pwd)/data/yolo11s-sar-512"
RUN_ROOT="runs/yolo26m-p2-dota-${TAG}"
mkdir -p "$RUN_ROOT"
DATA="$RUN_ROOT/dataset_all.yaml"  # both image folders train; validation still reports the (now seen) holdout
python - "$BASE/dataset.yaml" "$DATA" <<'PY'
import sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
names = source.read_text().split('names:', 1)[1]
target.write_text(f'path: {source.parent}\ntrain:\n  - images/train\n  - images/val\nval: images/val\nnc: 9\nnames:{names}')
PY

# ---- 100 epochs, as sbatch/yolo26_pretrained_full.sbatch stage 1 (job 330705) ----
python -u models/yolo/train_yolo26s_p2.py \
    --scale m --weights "$WEIGHTS" --train-output "$RUN_ROOT" --data "$DATA" \
    --epochs 100 --batch 16 --workers 4 --flipud 0.0 --close-mosaic 10
SOURCE="$RUN_ROOT/train/weights/last.pt"

# ---- the four 40-epoch stages, each from the previous one's last epoch ----
stage() {  # name, then finetune_detector options
    local name="$1"; shift
    python -u models/yolo/finetune_detector.py --tag "ft40-${name}-${TAG}" --source "$SOURCE" \
        --classifier runs/memorise-lr2-332326-package/image_classifier.pt \
        --all-data --skip-evaluation --seed 0 "$@"
    SOURCE="runs/ft40-${name}-${TAG}/train/weights/last.pt"
}
stage plain --epochs 40 --lr0 0.002 --speckle-prob 0
stage speckle --epochs 40 --lr0 0.001 --speckle-prob 0.5
stage third --epochs 40 --lr0 0.001
stage memorise-lr2 --epochs 40 --lr0 0.002 --no-augment
echo "DETECTOR_READY $SOURCE"

# ---- classifier, (optimistic) holdout scores and the official-validation ZIP ----
python -u models/yolo/package_candidate.py --tag "$TAG" --detector "$SOURCE" \
    --curves "runs/ft40-memorise-lr2-${TAG}/train/results.csv" \
    --box-audit results/memorise-lr2-332326/box_bias_crossfit.json --classifier-all-data
echo "ALLDATA_COMPLETE $TAG"
