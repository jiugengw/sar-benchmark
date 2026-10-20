#!/bin/bash
# Cascade R-CNN R50-FPN (Detectron2, COCO weights) on our SAR split; holdout predictions for official_mix_score.py.
# One-time setup on the login node (Detectron2 outside the project venv, so other jobs are untouched):
#   git clone --depth 1 https://github.com/facebookresearch/detectron2.git ~/scratchDirectory/detectron2-src
#   pip install --no-deps --target ~/scratchDirectory/cascade-libs fvcore iopath yacs tabulate portalocker "setuptools<81"
#   (cd ~/scratchDirectory/detectron2-src && pip install --no-deps --no-build-isolation --target ../cascade-libs .)
# Submit from the project root:
#   sbatch sbatch/cascade_rcnn.sbatch cascade-r50 --iters 20000      (or --smoke for a 50-iteration check)
#SBATCH --job-name=sar-cascade
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/sar-cascade-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${HOME}/scratchDirectory/cascade-libs${PYTHONPATH:+:${PYTHONPATH}}"

TAG="${1:?Give a run name}-${SLURM_JOB_ID}"
shift
python -u models/cascade/train.py --tag "$TAG" "$@"
