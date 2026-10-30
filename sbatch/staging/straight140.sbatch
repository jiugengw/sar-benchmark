#!/bin/bash
#SBATCH --job-name=sar-s1140
#SBATCH --partition=student
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/s1140-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python -u models/yolo/train_yolo26s_p2.py --scale m --weights pretrained/yolo26m-obb.pt --train-output runs/yolo26m-p2-dota-s1-140-${LR0:-auto}-$SLURM_JOB_ID --epochs 140 --batch 16 --workers 4 --flipud 0.0 --close-mosaic 10 --stem-stride 1 ${LR0:+--lr0 $LR0}
echo S1140_DONE
