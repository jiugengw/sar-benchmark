#!/bin/bash
# YOLO26m-P2, 100 epochs, starting from Ultralytics' DOTA weights (yolo26m-obb.pt: optical aerial photos with bridges,
# harbors, sports fields, storage tanks, ships, planes, vehicles) instead of COCO. Every setting matches the COCO-start
# run 328339, so the starting weights are the only change. TA-confirmed: optical aerial pretrained checkpoints are fine.
#
# Once, on the login node (compute nodes are offline):
#   curl -L --fail -o pretrained/yolo26m-obb.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-obb.pt
# Then: sbatch sbatch/yolo26m_p2_dota.sbatch
#SBATCH --job-name=sar-yolo26m-dota
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
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

WEIGHTS=pretrained/yolo26m-obb.pt
echo "23e0630f66857cf4b87535f6e705b065f1e8a33603640b8e61ace85b75312903  $WEIGHTS" | sha256sum --check
NAME="yolo26m-100-p2-dota-${SLURM_JOB_ID}"
RUN_ROOT="runs/$NAME"
RESULTS="results/$NAME"
export MPLCONFIGDIR="/tmp/$NAME"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$RUN_ROOT" "$RESULTS"

# Same as job 328339: 100 epochs, batch 16, Ultralytics defaults (no vertical flips, mosaic off for the last 10).
python -u models/yolo/train_yolo26s_p2.py \
    --scale m --weights "$WEIGHTS" --train-output "$RUN_ROOT" \
    --epochs 100 --batch 16 --workers 4 --flipud 0.0 --close-mosaic 10

CURVES="results/training-curves/$NAME"  # runs/ is not committed; keep the per-epoch history
mkdir -p "$CURVES"
cp "$RUN_ROOT/train/results.csv" "$CURVES/training.csv"
cp "$RUN_ROOT/train/args.yaml" "$CURVES/args.yaml"
cp "$RUN_ROOT/benchmark_settings.json" "$CURVES/benchmark_settings.json"

# Image classifier on the frozen detector (the 1024-neuron recipe), then both heads through the submission code.
python -u models/yolo/train_image_head.py \
    --detector-checkpoint "$RUN_ROOT/train/weights/best.pt" \
    --data data/course-data --dataset-root data/yolo11s-sar-512 \
    --split results/yolo26m-100-p2-328339/seed_0/split.json \
    --output "$RUN_ROOT/image_classifier.pt" \
    --feature-pooling spatial --hidden-dim 1024 --dropout 0.35 --learning-rate 0.0005 \
    --weight-decay 0.001 --train-flips --head-batch 256 --epochs 240 --batch 32 --workers 4
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
for BRANCH in one2many one2one; do
    python - "$RUN_ROOT/train/weights/best.pt" "$RUN_ROOT/image_classifier.pt" "$BRANCH" "$RESULTS/$BRANCH" <<'PY'
import sys
from pathlib import Path
from models.yolo.deadline_recovery import evaluate_export
evaluate_export(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
PY
    echo "== DOTA start, $BRANCH head"
    python -m sarbench.twins score "$RESULTS/$BRANCH"
done

# Side by side with the COCO start: best holdout mAP during training (same Ultralytics validation for both runs),
# and per-class AP of the exported one-to-many predictions.
python - "$CURVES/training.csv" "$RESULTS/one2many/metrics.json" <<'PY'
import csv, json, sys
def best(path):
    rows = list(csv.DictReader(open(path)))
    key = next(k for k in rows[0] if k.strip() == 'metrics/mAP50-95(B)')
    return max(float(r[key]) for r in rows)
print(f"best training-time holdout mAP: COCO start {best('results/training-curves/yolo26m-100-p2-328339/training.csv'):.4f}"
      f"  DOTA start {best(sys.argv[1]):.4f}")
coco = json.load(open('results/yolo26m-100-p2-328339/seed_0/metrics.json'))['val']['per_class_AP']
dota = json.load(open(sys.argv[2]))['val']['per_class_AP']
print(f"{'class':14s} {'COCO':>7s} {'DOTA':>7s}   (COCO: results/yolo26m-100-p2-328339/seed_0)")
for name in coco:
    print(f"{name:14s} {coco[name]:7.4f} {dota.get(name, float('nan')):7.4f}")
PY

PACKAGE_DIR="nonsubmissions/$(date +%d%m%y)/$NAME"
PACKAGE="$PACKAGE_DIR/yolo26m_100_p2_dota_${SLURM_JOB_ID}"
mkdir -p "$PACKAGE_DIR"
python -u models/yolo/make_submission.py \
    --weights "$RUN_ROOT/train/weights/best.pt" \
    --classifier-checkpoint "$RUN_ROOT/image_classifier.pt" \
    --detection-branch one2many \
    --data data/course-data --classes data/yolo11s-sar-512/conversion_manifest.json \
    --out "$PACKAGE.zip" --batch 16 --workers 4
cp "$RESULTS/one2many/metrics.json" "$PACKAGE_DIR/holdout.metrics.json"
printf '{"zip": "%s", "histories": [{"type": "ultralytics", "path": "%s/training.csv", "label": "YOLO26m-P2 from DOTA weights, 100 epochs"}]}\n' \
    "$(basename "$PACKAGE").zip" "$CURVES" > "$PACKAGE.run.json"
echo "Package: $PACKAGE.zip"
