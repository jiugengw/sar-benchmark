#!/bin/bash
# Image classifier stacked with detection summaries (models/yolo/classifier_stack_study.py): export the holdout, then cross-fit.
#   sbatch sbatch/classifier_stack_study.sbatch runs/ft40-memorise-lr2-332326/train/weights/best.pt runs/memorise-lr2-332326-package/image_classifier.pt stack-memorise-lr2
#SBATCH --job-name=sar-stack
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
export MPLCONFIGDIR="/tmp/sar-stack-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

python - "${1:?detector weights}" "${2:?image classifier}" "${3:?tag}" <<'PY'
import sys
from pathlib import Path
from models.yolo.deadline_recovery import evaluate_export
weights, classifier, tag = sys.argv[1:]
evaluate_export(Path(weights), Path(classifier), 'one2many', Path('results') / tag / 'one2many')
PY
python -u models/yolo/classifier_stack_study.py --results "results/${3}/one2many"
