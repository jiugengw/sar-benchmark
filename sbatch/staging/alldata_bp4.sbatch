#!/bin/bash
# Build a submission candidate from a trained detector: 1024-neuron classifier, holdout scores, box nudge, ZIP.
# Submit from the project directory: name, detector weights, and (optional) its training.csv, e.g.:
#   sbatch sbatch/package_candidate.sbatch speckle-cand runs/ft40-speckle-330447/train/weights/best.pt \
#       results/training-curves/ft40-speckle-330447/training.csv
# For a detector trained on the holdout too (all-data runs), reuse a nudge fitted on a held-out model:
#   BOX_AUDIT=results/third-stage-331789/box_bias_crossfit.json sbatch sbatch/package_candidate.sbatch ...
#SBATCH --job-name=sar-alld-bp4
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
export MPLCONFIGDIR="/tmp/sar-package-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

TAG="alldata-bp4-${SLURM_JOB_ID}"
CLS=runs/memorise-lr2-332326-package/image_classifier.pt
python -u models/yolo/finetune_detector.py --tag "ft40-third-$TAG" --source runs/ft40-speckle-alldata-332975/train/weights/last.pt --classifier $CLS --all-data --skip-evaluation --seed 0 --epochs 40 --lr0 0.001 --repeat-classes airport,harbor,oil_tank,wind_turbine,bridge:4,playground:4 --repeat-factor 2 --repeat-width 512
python -u models/yolo/finetune_detector.py --tag "ft40-memorise-lr2-$TAG" --source "runs/ft40-third-$TAG/train/weights/last.pt" --classifier $CLS --all-data --skip-evaluation --seed 0 --epochs 40 --lr0 0.002 --no-augment --tal-beta 8
python -u models/yolo/package_candidate.py --tag "$TAG" --detector "runs/ft40-memorise-lr2-$TAG/train/weights/last.pt" --curves "runs/ft40-memorise-lr2-$TAG/train/results.csv" --box-audit results/memorise-lr2-332326/box_bias_crossfit.json --classifier-all-data
echo "ALLDATA_BP4_COMPLETE $TAG"
