#!/bin/bash
#SBATCH --job-name=sar-polfull
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/polfull-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
V=runs/pol-views-335767
T=bp4-polviews-third-$SLURM_JOB_ID
python -u models/yolo/train_with_extra.py --extra-zip $V/views.zip --extra-labels $V/views_labels.pkl --tag $T --source runs/chain-dota-330938-speckle/train/weights/best.pt --classifier runs/chain-dota-330938-package/image_classifier.pt --skip-evaluation --lr0 0.001 --repeat-classes airport,harbor,oil_tank,wind_turbine,bridge:4,playground:4 --repeat-factor 2 --repeat-width 512
python -u models/yolo/train_with_extra.py --extra-zip $V/views.zip --extra-labels $V/views_labels.pkl --tag bp4-polviews-full-memorise-$SLURM_JOB_ID --source runs/$T/train/weights/best.pt --classifier runs/third-stage-331789-package/image_classifier.pt --no-augment --lr0 0.002 --tal-beta 8
echo POLFULL_COMPLETE
