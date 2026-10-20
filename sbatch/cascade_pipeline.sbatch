#!/bin/bash
# Cascade R-CNN in two phases, like our YOLO recipe, in one job:
#   variety:  ~100 epochs (52,800 steps) from COCO weights, scale jitter 0.5-1.5x + crop/pad to 512 + flips
#   memorise: ~40 epochs (21,000 steps) from the variety phase, no augmentation, learning rate 10x lower
# Holdout predictions of both phases land in results/<tag>-variety and results/<tag>-memorise.
#   sbatch sbatch/cascade_pipeline.sbatch cascade-pipeline
#SBATCH --job-name=sar-casc-pipe
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/sar-casc-pipe-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${HOME}/scratchDirectory/cascade-libs${PYTHONPATH:+:${PYTHONPATH}}"

TAG="${1:?Give a run name}-${SLURM_JOB_ID}"
python -u models/cascade/train.py --tag "${TAG}-variety" --iters 52800 --clean-iters 0 --eval-period 8000 \
    --scale-jitter 0.5 1.5
python -u models/cascade/train.py --tag "${TAG}-memorise" --iters 21000 --clean-iters 0 --eval-period 7000 \
    --no-augment --base-lr 0.002 --weights "runs/${TAG}-variety/model_final.pth"
echo "CASCADE_PIPELINE_COMPLETE $TAG"
