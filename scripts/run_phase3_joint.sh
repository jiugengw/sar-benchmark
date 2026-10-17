#!/usr/bin/env bash
# Run both Phase 3 joint fine-tunes in the current interactive GPU session.
# Activate a CUDA-enabled Python environment before running this script.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! git lfs version >/dev/null 2>&1; then
    echo "Git LFS is required to fetch the E5/E7 checkpoints and course data." >&2
    exit 2
fi

git lfs pull --include="data/*.tar.gz,models/yolo/phase3_assets/checkpoints/*.pt"

for checkpoint in \
    models/yolo/phase3_assets/checkpoints/E5_yolo26m_p2_best.pt \
    models/yolo/phase3_assets/checkpoints/E5_yolo26m_image_classifier.pt \
    models/yolo/phase3_assets/checkpoints/E7_rtdetr_l_p2_best.pt \
    models/yolo/phase3_assets/checkpoints/E7_rtdetr_l_image_classifier.pt; do
    if [[ ! -s "$checkpoint" ]]; then
        echo "Missing or empty Phase 3 checkpoint: $checkpoint" >&2
        exit 2
    fi
done

if [[ ! -s data/train.tar.gz || ! -s data/val.tar.gz ]]; then
    echo "Course data archives are missing after Git LFS pull." >&2
    exit 2
fi

python - <<'PY'
import torch
import ultralytics
import transformers

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is visible. Run this from the interactive GPU session.")
print(f"PyTorch {torch.__version__}; Ultralytics {ultralytics.__version__}; Transformers {transformers.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}; visible memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
PY

mkdir -p data/course-data
if [[ ! -s data/course-data/train/labels.csv || ! -s data/course-data/val/images.json ]]; then
    tar -xzf data/train.tar.gz -C data/course-data
    tar -xzf data/val.tar.gz -C data/course-data
fi

python -u models/yolo/prepare_dataset.py \
    --data data/course-data \
    --split results/yolo26m-100-p2-328339/seed_0/split.json \
    --out data/yolo11s-sar-512 \
    --size 512

JOB_TAG="${PHASE3_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
YOLO_OUTPUT="runs/phase3-yolo26m-joint-${JOB_TAG}"
RTDETR_OUTPUT="runs/phase3-rtdetr-l-joint-${JOB_TAG}"
mkdir -p "$YOLO_OUTPUT" "$RTDETR_OUTPUT"

python -u models/yolo/preflight_rtdetr_joint.py \
    --config models/yolo/experiment_configs/phase3_rtdetr_l.yaml \
    --output "${RTDETR_OUTPUT}/preflight.json"

python -u models/yolo/train_joint.py \
    --config models/yolo/experiment_configs/phase3_yolo26m.yaml \
    --experiment E2 \
    --output "$YOLO_OUTPUT"

python -u models/yolo/train_rtdetr_joint.py \
    --config models/yolo/experiment_configs/phase3_rtdetr_l.yaml \
    --experiment E2 \
    --output "$RTDETR_OUTPUT"

echo "Phase 3 joint fine-tuning finished."
echo "YOLO26m run: $YOLO_OUTPUT"
echo "RT-DETR-L run: $RTDETR_OUTPUT"
