#!/bin/bash
#SBATCH --job-name=sar-s1heads
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/s1h-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
D=runs/ft40-s1r-memorise-337666/train/weights/best.pt
O=runs/s1-heads-$SLURM_JOB_ID
python -u models/yolo/branch_head_study.py --detector $D --output $O/branch_head.pt --epochs 30 --batch 16
python -u models/yolo/conv_head_study.py --detector $D --output $O/conv_head.pt --batch 16
echo S1_HEADS_DONE
