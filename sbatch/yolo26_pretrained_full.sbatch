#!/bin/bash
# One job, start to finish: YOLO26-P2 from pretrained weights for 100 epochs (classifier, holdout scores, ZIP),
# then the speckle-cand chain on top (+40 plain, +40 speckle, classifier refit, box nudge, ZIP).
# Both stages leave their own scores and ZIP, so the 100-epoch result is not lost if the chain is ignored.
#   sbatch sbatch/yolo26_pretrained_full.sbatch dota l      (DOTA weights: pretrained/yolo26<scale>-obb.pt)
#   sbatch sbatch/yolo26_pretrained_full.sbatch coco l      (COCO weights: pretrained/yolo26<scale>.pt)
# Input-filter variant (experiment B): FILTER=water sbatch sbatch/yolo26_pretrained_full.sbatch dota m
#   builds data/yolo-sar-512-<filter> once, trains and predicts on [SAR, despeckled, water map] (see input_filters.py)
# The 100-epoch stage repeats sbatch/yolo26m_pretraining_compare.sbatch with the scale as a parameter.
#SBATCH --job-name=sar-full
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
export MPLCONFIGDIR="/tmp/sar-full-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

INITIALIZATION="${1:?dota or coco}"
SCALE="${2:?m or l}"
case "$INITIALIZATION:$SCALE" in
    dota:m) WEIGHTS=pretrained/yolo26m-obb.pt; SHA=23e0630f66857cf4b87535f6e705b065f1e8a33603640b8e61ace85b75312903 ;;
    dota:l) WEIGHTS=pretrained/yolo26l-obb.pt; SHA=8674b0c24bf68aab5eb45009e0ac3808ce432237edf8cb5c50ae2191cb263a2b ;;
    coco:m) WEIGHTS=pretrained/yolo26m.pt; SHA= ;;
    coco:l) WEIGHTS=pretrained/yolo26l.pt; SHA=9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26 ;;
    *) echo "Unsupported: $INITIALIZATION $SCALE" >&2; exit 2 ;;
esac
[[ -z "$SHA" ]] || echo "$SHA  $WEIGHTS" | sha256sum --check
FILTER="${FILTER:-none}"
DATA_ARGS=()
CHAIN_DATA_ARGS=()
SUFFIX=""
if [[ "$FILTER" != none ]]; then
    export SAR_INPUT_FILTER="$FILTER"  # read by make_submission.py and train_image_head.py at prediction time
    DATA="data/yolo-sar-512-${FILTER}"
    if [[ ! -f "$DATA/dataset.yaml" ]]; then
        python -u models/yolo/build_variant_dataset.py --out "$DATA" --filter "$FILTER"
    fi
    DATA_ARGS=(--data "$DATA/dataset.yaml" --no-colour-jitter)
    CHAIN_DATA_ARGS=(--data "$DATA/dataset.yaml")
    SUFFIX="-${FILTER}"
fi
TAG="yolo26${SCALE}-p2-${INITIALIZATION}${SUFFIX}-verified-${SLURM_JOB_ID}"
RUN_ROOT="runs/$TAG"
RESULTS="results/$TAG"
CURVES="results/training-curves/$TAG"
mkdir -p "$RUN_ROOT" "$RESULTS" "$CURVES"

# ---- Stage 1: 100 epochs, identical to the verified m pretraining runs except for the scale ----
python -u models/yolo/train_yolo26s_p2.py \
    --scale "$SCALE" --weights "$WEIGHTS" --train-output "$RUN_ROOT" \
    --epochs 100 --batch 16 --workers 4 --flipud 0.0 --close-mosaic 10 "${DATA_ARGS[@]}"
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
    print(f'== 100 epochs, {branch}', flush=True)
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
printf '{"zip": "%s.zip", "histories": [{"type": "ultralytics", "path": "%s/training.csv", "label": "%s"}], "note": "Verified pretrained start; 100 epochs; no continuation or box nudge", "official_score": null}\n' \
    "$TAG" "$CURVES" "$TAG" > "$PACKAGE_DIR/$TAG.run.json"
echo "STAGE1_COMPLETE $PACKAGE_DIR/$TAG.zip"

# ---- Stage 2: the speckle-cand chain on top ----
python -u models/yolo/continue_chain.py --tag "chain-${INITIALIZATION}-${SCALE}${SUFFIX}-${SLURM_JOB_ID}" \
    --source "$RUN_ROOT/train/weights/best.pt" --classifier "$RUN_ROOT/image_classifier.pt" "${CHAIN_DATA_ARGS[@]}"
echo "FULL_PIPELINE_COMPLETE $TAG"
