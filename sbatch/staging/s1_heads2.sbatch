#!/bin/bash
#SBATCH --job-name=sar-s1heads2
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/s1h2-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
D=runs/ft40-s1r-memorise-337666/train/weights/best.pt; O=runs/s1-heads2-$SLURM_JOB_ID; mkdir -p $O
python -u models/yolo/train_image_head.py --detector-checkpoint $D --output $O/image_classifier.pt --epochs 240 --batch 32 --head-batch 256 --workers 4 --feature-pooling spatial --hidden-dim 1024 --dropout 0.35 --learning-rate 0.0005 --weight-decay 0.001 --train-flips
python -u models/yolo/head_probs.py --detector $D --classifier $O/image_classifier.pt --out $O/meanhead_holdout_probs.json
python -u models/yolo/branch_head_study.py --detector $D --output $O/branch_seed1.pt --epochs 30 --batch 16 --seed 1
echo S1_HEADS2_DONE
