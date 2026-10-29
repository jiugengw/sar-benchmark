#!/bin/bash
#SBATCH --job-name=sar-polviews
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/polviews-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
OUT=runs/pol-views-$SLURM_JOB_ID
python -u models/yolo/pol_translate.py --out $OUT
python -u models/yolo/train_with_extra.py --extra-zip $OUT/views.zip --extra-labels $OUT/views_labels.pkl --tag bp4-polviews-memorise-$SLURM_JOB_ID --source runs/ft40-m4rep-bp4-third-335138/train/weights/best.pt --classifier runs/third-stage-331789-package/image_classifier.pt --no-augment --lr0 0.002 --tal-beta 8
echo POLVIEWS_COMPLETE
