#!/bin/bash
# Corrected matched comparison. Task 0 = DOTA; task 1 = COCO.
# Existing matched-large job keeps its allocation; these tasks use available slots.
#SBATCH --job-name=sar-m-pretrain
#SBATCH --array=0-1%2
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x-%A_%a.out

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
export MPLCONFIGDIR="/tmp/sar-m-pretrain-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) INITIALIZATION=dota; WEIGHTS=pretrained/yolo26m-obb.pt
       echo "23e0630f66857cf4b87535f6e705b065f1e8a33603640b8e61ace85b75312903  $WEIGHTS" | sha256sum --check ;;
    1) INITIALIZATION=coco; WEIGHTS=pretrained/yolo26m.pt ;;
    *) echo 'Unexpected array task' >&2; exit 1 ;;
esac
TAG="yolo26m-p2-${INITIALIZATION}-verified-${SLURM_ARRAY_JOB_ID}"
RUN_ROOT="runs/$TAG"
RESULTS="results/$TAG"
CURVES="results/training-curves/$TAG"
mkdir -p "$RUN_ROOT" "$RESULTS" "$CURVES"

python -u models/yolo/train_yolo26s_p2.py \
    --scale m --weights "$WEIGHTS" --train-output "$RUN_ROOT" \
    --epochs 100 --batch 16 --workers 4 --flipud 0.0 --close-mosaic 10
cp "$RUN_ROOT/train/results.csv" "$CURVES/training.csv"
cp "$RUN_ROOT/train/args.yaml" "$CURVES/args.yaml"
cp "$RUN_ROOT/benchmark_settings.json" "$CURVES/benchmark_settings.json"
cp "$RUN_ROOT/initialization_audit.json" "$CURVES/initialization_audit.json"

python -u models/yolo/train_image_head.py \
    --detector-checkpoint "$RUN_ROOT/train/weights/best.pt" \
    --data data/course-data --dataset-root data/yolo11s-sar-512 \
    --split results/yolo26m-100-p2-328339/seed_0/split.json \
    --output "$RUN_ROOT/image_classifier.pt" \
    --feature-pooling spatial --hidden-dim 1024 --dropout 0.35 --learning-rate 0.0005 \
    --weight-decay 0.001 --train-flips --head-batch 256 --epochs 240 --batch 32 --workers 4
cp "$RUN_ROOT/image_classifier.metrics.json" "$CURVES/classifier.metrics.json"
cp "$RUN_ROOT/image_classifier.training.csv" "$CURVES/classifier_training.csv"

python - "$RUN_ROOT" "$RESULTS" <<'PY'
import json, sys
from pathlib import Path
from models.yolo.deadline_recovery import evaluate_export
from sarbench.twins import score
work, output = map(Path, sys.argv[1:])
report = {}
for branch in ('one2many', 'one2one'):
    evaluate_export(work / 'train/weights/best.pt', work / 'image_classifier.pt', branch, output / branch)
    report[branch] = score(output / branch, Path('data/course-data'))
(output / 'full_clean_scores.json').write_text(json.dumps(report, indent=2) + '\n')
PY

PACKAGE_DIR="nonsubmissions/$(date +%d%m%y)/$TAG"
mkdir -p "$PACKAGE_DIR"
python -u models/yolo/make_submission.py \
    --weights "$RUN_ROOT/train/weights/best.pt" --classifier-checkpoint "$RUN_ROOT/image_classifier.pt" \
    --detection-branch one2many --out "$PACKAGE_DIR/$TAG.zip" --batch 16 --workers 4
cp "$RESULTS/one2many/metrics.json" "$PACKAGE_DIR/holdout.metrics.json"
cp "$RESULTS/full_clean_scores.json" "$PACKAGE_DIR/full_clean_scores.json"
cp "$RUN_ROOT/initialization_audit.json" "$PACKAGE_DIR/initialization_audit.json"
python - "$PACKAGE_DIR" "$TAG" "$CURVES" <<'PY'
import json, sys
from pathlib import Path
output, tag, curves = sys.argv[1:]
(Path(output) / f'{tag}.run.json').write_text(json.dumps({
    'zip': f'{tag}.zip', 'histories': [{'type': 'ultralytics', 'path': f'{curves}/training.csv', 'label': tag},
    {'type': 'classifier_csv', 'path': f'{curves}/classifier_training.csv', 'label': 'Identical 1024-unit refit'}],
    'note': 'Verified pretrained initialization; 100 epochs; no continuation, speckle, box correction or TTA',
    'official_score': None}, indent=2) + '\n')
PY
echo "VERIFIED_PRETRAINING_COMPLETE $TAG"
