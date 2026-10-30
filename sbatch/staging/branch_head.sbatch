#!/bin/bash
#SBATCH --job-name=sar-branchcls
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/branch-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python -u models/yolo/branch_head_study.py --detector runs/soup-bp4/six.pt --output runs/branch-head-six-$SLURM_JOB_ID/branch_head.pt
echo BRANCH_DONE
