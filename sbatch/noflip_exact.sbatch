#!/bin/bash
# No flips anywhere: the 100-epoch YOLO26m-P2 stage from DOTA weights with left-right flips off (up-down flips are
# already off), then exact 80 (80 epochs, no augmentation, lr 0.002) on top. The only difference from exact80
# (332471, which starts from the flipped runs/yolo26m-p2-dota-verified-330705) is the flips in the first 100 epochs:
# radar looks from one side, so shadow and layover always fall on a fixed side, and a flipped image is not physical.
#   sbatch sbatch/noflip_exact.sbatch
#SBATCH --job-name=sar-noflip
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=11:00:00
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
export MPLCONFIGDIR="/tmp/sar-noflip-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

WEIGHTS=pretrained/yolo26m-obb.pt
echo "23e0630f66857cf4b87535f6e705b065f1e8a33603640b8e61ace85b75312903  $WEIGHTS" | sha256sum --check
TAG="yolo26m-p2-dota-noflip-${SLURM_JOB_ID}"
RUN_ROOT="runs/$TAG"
RESULTS="results/$TAG"
CURVES="results/training-curves/$TAG"
mkdir -p "$RUN_ROOT" "$RESULTS" "$CURVES"

# ---- Stage 1: 100 epochs, as in sbatch/yolo26_pretrained_full.sbatch but with --fliplr 0.0 ----
python -u models/yolo/train_yolo26s_p2.py \
    --scale m --weights "$WEIGHTS" --train-output "$RUN_ROOT" \
    --epochs 100 --batch 16 --workers 4 --flipud 0.0 --fliplr 0.0 --close-mosaic 10
cp "$RUN_ROOT/train/results.csv" "$CURVES/training.csv"
cp "$RUN_ROOT/train/args.yaml" "$CURVES/args.yaml"
cp "$RUN_ROOT/benchmark_settings.json" "$CURVES/benchmark_settings.json"

python -u models/yolo/train_image_head.py \
    --detector-checkpoint "$RUN_ROOT/train/weights/best.pt" \
    --data data/course-data --dataset-root data/yolo11s-sar-512 \
    --split results/yolo26m-100-p2-328339/seed_0/split.json \
    --output "$RUN_ROOT/image_classifier.pt" \
    --feature-pooling spatial --hidden-dim 1024 --dropout 0.35 --learning-rate 0.0005 \
    --weight-decay 0.001 --train-flips --head-batch 256 --epochs 240 --batch 32 --workers 4

python - "$RUN_ROOT" "$RESULTS" <<'PY'
import json, sys
from pathlib import Path
from models.yolo.deadline_recovery import evaluate_export
from sarbench.twins import score
work, output = map(Path, sys.argv[1:])
evaluate_export(work / 'train/weights/best.pt', work / 'image_classifier.pt', 'one2many', output / 'one2many')
print('== 100 epochs without flips, one2many', flush=True)
report = {'one2many': score(output / 'one2many', Path('data/course-data'))}
(output / 'full_clean_scores.json').write_text(json.dumps(report, indent=2) + '\n')
PY
echo "STAGE1_COMPLETE $TAG"

# ---- Stage 2: exact 80 on top, the same command as job 332471 ----
python -u models/yolo/finetune_detector.py --tag "ft40-exact80-noflip-${SLURM_JOB_ID}" \
    --source "$RUN_ROOT/train/weights/best.pt" --classifier "$RUN_ROOT/image_classifier.pt" \
    --no-augment --lr0 0.002 --epochs 80
echo "NOFLIP_COMPLETE $TAG"
