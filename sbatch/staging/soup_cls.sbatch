#!/bin/bash
# both classifier heads on a soup detector: the packaged spatial head, then holdout probabilities; the conv branch
#SBATCH --job-name=sar-soupcls
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/soupcls-$SLURM_JOB_ID PYTHONUNBUFFERED=1
DET=$1; NAME=$2
python -u models/yolo/train_image_head.py --detector-checkpoint $DET --output runs/soup-bp4/$NAME-meanhead.pt --epochs 240 --batch 32 --head-batch 256 --workers 4 --feature-pooling spatial --hidden-dim 1024 --dropout 0.35 --learning-rate 0.0005 --weight-decay 0.001 --train-flips
python -u models/yolo/head_probs.py --detector $DET --classifier runs/soup-bp4/$NAME-meanhead.pt --out runs/soup-bp4/$NAME-meanhead_holdout_probs.json
python -u models/yolo/conv_head_study.py --detector $DET --output runs/soup-bp4/$NAME-conv/conv_head.pt --epochs 25
python -u models/yolo/classifier_combo_study.py --mean runs/soup-bp4/$NAME-meanhead_holdout_probs.json --conv runs/soup-bp4/$NAME-conv/conv_head.holdout_probs.npy
echo SOUPCLS_COMPLETE
