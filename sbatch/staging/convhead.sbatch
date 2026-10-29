#!/bin/bash
#SBATCH --job-name=sar-convhead
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/convhead-$SLURM_JOB_ID PYTHONUNBUFFERED=1
python -u models/yolo/conv_head_study.py --detector runs/ft40-m4rep-bp4-memorise-335138/train/weights/best.pt --output runs/conv-head-bp4-$SLURM_JOB_ID/conv_head.pt "$@"
