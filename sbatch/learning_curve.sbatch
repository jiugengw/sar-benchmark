#!/bin/bash
# Learning curve: the 100-epoch YOLO26m-P2 stage from DOTA weights on a share of the training images, everything else
# as runs/yolo26m-p2-dota-verified-330705 (all 8,453 images, clean holdout 41.09). If an object class's holdout AP
# drops clearly with less data, that class is still limited by data (more data or repeats would help); if it barely
# moves, more data won't help it.
#   sbatch sbatch/learning_curve.sbatch 0.5
#SBATCH --job-name=sar-curve
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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
export MPLCONFIGDIR="/tmp/sar-curve-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

FRACTION="${1:?share of the training images, e.g. 0.5}"
WEIGHTS=pretrained/yolo26m-obb.pt
echo "23e0630f66857cf4b87535f6e705b065f1e8a33603640b8e61ace85b75312903  $WEIGHTS" | sha256sum --check
TAG="yolo26m-p2-dota-train${FRACTION}-${SLURM_JOB_ID}"
DATA="data/yolo-sar-512-train${FRACTION}-${SLURM_JOB_ID}"  # hard links; deleted at the end (cluster file limit)
RUN_ROOT="runs/$TAG"
RESULTS="results/$TAG"
mkdir -p "$RUN_ROOT" "$RESULTS"
trap 'rm -rf "$DATA"' EXIT

python -u models/yolo/build_variant_dataset.py --out "$DATA" --train-fraction "$FRACTION"
python -u models/yolo/train_yolo26s_p2.py \
    --scale m --weights "$WEIGHTS" --train-output "$RUN_ROOT" --data "$DATA/dataset.yaml" \
    --epochs 100 --batch 16 --workers 4 --flipud 0.0 --close-mosaic 10

# The image classifier only feeds F1, not box AP; it is trained as for 330705 so the scoring code runs unchanged
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
report = {'one2many': score(output / 'one2many', Path('data/course-data'))}
(output / 'full_clean_scores.json').write_text(json.dumps(report, indent=2) + '\n')
base = json.loads(Path('results/yolo26m-p2-dota-verified-330705/full_clean_scores.json').read_text())['one2many']['clean']
mine = report['one2many']['clean']
print(f"{'clean holdout AP':16s} {'all data':>8s} {'this share':>10s}")
for name in ('mAP', *base['per_class_AP']):
    a = base['mAP'] if name == 'mAP' else base['per_class_AP'][name]
    b = mine['mAP'] if name == 'mAP' else mine['per_class_AP'][name]
    print(f'{name:16s} {100 * a:8.2f} {100 * b:10.2f}')
PY
echo "CURVE_COMPLETE $TAG"
