#!/bin/bash
#SBATCH --job-name=sar-yolo11n-p2-package
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/yolo11n-p2-package-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}/yolo_baseline${PYTHONPATH:+:${PYTHONPATH}}"

mapfile -t RUN_CANDIDATES < <(find runs -maxdepth 1 -type d -name 'yolo11n-100-p2-*' -printf '%T@ %p\n' | sort -nr)
if [[ "${#RUN_CANDIDATES[@]}" -eq 0 ]]; then
    echo "No completed YOLO11n-P2 run directory found" >&2
    exit 1
fi
RUN_ROOT="${RUN_CANDIDATES[0]#* }"
RUN_NAME="${RUN_ROOT##*/}"
JOB_SUFFIX="${RUN_NAME##*-}"
RESULT_ROOT="results/${RUN_NAME}/seed_0"
PACKAGE_DIR="submissions/161026/tests-no-submit/${RUN_NAME}"
CHECKPOINT="$RUN_ROOT/train/weights/best.pt"
CLASSIFIER="$RUN_ROOT/image_classifier.pt"
if [[ ! -f "$CHECKPOINT" || ! -f "$CLASSIFIER" || ! -f "$RESULT_ROOT/metrics.json" ]]; then
    echo "Run artifacts are incomplete for ${RUN_NAME}" >&2
    exit 1
fi
mkdir -p "$PACKAGE_DIR/scoring_result"
touch "$PACKAGE_DIR/scoring_result/.gitkeep"
python -u yolo_baseline/make_submission.py \
    --weights "$CHECKPOINT" \
    --classifier-checkpoint "$CLASSIFIER" \
    --box-class-policy native_detector_classes \
    --data data/course-data \
    --classes data/yolo11s-sar-512/conversion_manifest.json \
    --out "$PACKAGE_DIR/yolo11n_100_p2_${JOB_SUFFIX}.zip" \
    --batch 16 --workers 4
cp "$RESULT_ROOT/metrics.json" "$PACKAGE_DIR/yolo11n_100_p2_${JOB_SUFFIX}.metrics.json"
