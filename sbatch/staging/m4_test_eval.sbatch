#!/bin/bash
# PRIVATE diagnostic: course-only detectors on unseen M4-SAR test images.
#SBATCH --job-name=sar-m4test
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/common/scratch/users/j/jgwong.2023/private-val/%x-%j.out
set -euo pipefail
cd /common/scratch/users/j/jgwong.2023/sar-benchmark
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MPLCONFIGDIR="/tmp/m4test-${SLURM_JOB_ID}"
export PYTHONPATH="$PWD:$PWD/models/yolo"
python -u ../private-val/m4_test_eval.py "$@"
