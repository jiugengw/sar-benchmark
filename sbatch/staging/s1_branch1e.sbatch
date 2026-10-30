#!/bin/bash
#SBATCH --job-name=sar-s1branch1e
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/s1b1e-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python -u models/yolo/branch_head_study.py --detector runs/ft40-s1r-memorise-337666/train/weights/best.pt --init-from runs/ft40-s1r-third-337119/train/weights/best.pt --output runs/s1-branch1e-$SLURM_JOB_ID/branch_head.pt --split 1 --epochs 30 --batch 8
echo S1_BRANCH1E_DONE
