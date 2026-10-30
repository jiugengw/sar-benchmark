#!/bin/bash
# Build a submission candidate from a trained detector: 1024-neuron classifier, holdout scores, box nudge, ZIP.
# Submit from the project directory: name, detector weights, and (optional) its training.csv, e.g.:
#   sbatch sbatch/package_candidate.sbatch speckle-cand runs/ft40-speckle-330447/train/weights/best.pt \
#       results/training-curves/ft40-speckle-330447/training.csv
# For a detector trained on the holdout too (all-data runs), reuse a nudge fitted on a held-out model:
#   BOX_AUDIT=results/third-stage-331789/box_bias_crossfit.json sbatch sbatch/package_candidate.sbatch ...
#SBATCH --job-name=sar-clstest
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/sar-package-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

python -u models/yolo/train_image_head.py --detector-checkpoint runs/ft40-memorise-lr2-332326/train/weights/best.pt --output /tmp/clstest-${SLURM_JOB_ID}/image_classifier.pt --split results/yolo26m-100-p2-328339/seed_0/split.json --epochs 2 --batch 32 --head-batch 256 --workers 4 --feature-pooling spatial --hidden-dim 1024 --dropout 0.35 --learning-rate 0.0005 --weight-decay 0.001 --train-flips --all-data && python -c "import torch; c=torch.load(\"/tmp/clstest-${SLURM_JOB_ID}/image_classifier.pt\", weights_only=False); print(\"CLASSIFIER_OK\", c[\"supervision\"], c[\"training\"][\"checkpoint_selection\"], c[\"training\"][\"best_epoch\"])"
