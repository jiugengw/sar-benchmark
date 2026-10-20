#!/bin/bash
# Training-label cleaning with out-of-fold predictions, then the control's last two stages on the cleaned labels:
#   1. the 100-epoch model on the complementary half of the training split (the learning-curve job 332768 trained the
#      other half), so every training image has a model that never saw it
#   2. models/yolo/relabel_train.py: remove labels that model cannot see at all, add confident missing objects
#   3. third stage + memorise lr 0.002 on the cleaned labels, from the DOTA chain's speckle stage (as the control)
# The holdout keeps its original labels, so the result compares directly with the control (clean 50.40, mix 51.80).
#   sbatch sbatch/relabel_pipeline.sbatch
#SBATCH --job-name=sar-relabel
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/sar-relabel-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

HALF_A=runs/yolo26m-p2-dota-train0.5-332768/train/weights/best.pt
HALF_B_RUN="runs/yolo26m-p2-dota-train0.5c-${SLURM_JOB_ID}"
HALF_DATA="data/yolo-sar-512-train0.5c-${SLURM_JOB_ID}"   # hard links, deleted after training (file limit)
CLEAN_DATA="data/yolo-sar-512-relabel-${SLURM_JOB_ID}"    # 8,453 label files, deleted at the end
trap 'rm -rf "$HALF_DATA" "$CLEAN_DATA"' EXIT

python -u models/yolo/build_variant_dataset.py --out "$HALF_DATA" --train-fraction 0.5 --complement
python -u models/yolo/train_yolo26s_p2.py --scale m --weights pretrained/yolo26m-obb.pt --train-output "$HALF_B_RUN" \
    --data "$HALF_DATA/dataset.yaml" --epochs 100 --batch 16 --workers 4 --flipud 0.0 --close-mosaic 10
rm -rf "$HALF_DATA"

python -u models/yolo/relabel_train.py --model-a "$HALF_A" --model-b "$HALF_B_RUN/train/weights/best.pt" \
    --out "$CLEAN_DATA"
mkdir -p "results/relabel-${SLURM_JOB_ID}" && cp "$CLEAN_DATA/relabel_report.json" "results/relabel-${SLURM_JOB_ID}/"

THIRD="ft40-relabel-third-${SLURM_JOB_ID}"
python -u models/yolo/finetune_detector.py --tag "$THIRD" --data "$CLEAN_DATA/dataset.yaml" \
    --source runs/chain-dota-330938-speckle/train/weights/best.pt \
    --classifier runs/chain-dota-330938-package/image_classifier.pt --skip-evaluation
python -u models/yolo/finetune_detector.py --tag "ft40-relabel-memorise-${SLURM_JOB_ID}" --data "$CLEAN_DATA/dataset.yaml" \
    --source "runs/$THIRD/train/weights/best.pt" \
    --classifier runs/third-stage-331789-package/image_classifier.pt --no-augment --lr0 0.002
echo "RELABEL_PIPELINE_COMPLETE"
