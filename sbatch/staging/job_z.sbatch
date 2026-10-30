#!/bin/bash
# job_z: rebuild the final ZIP from the saved model with models/yolo/make_final_zip.py (uploaded as
# ~/sbatch-staging/job_z.py, a neutral name) and compare it with the uploaded +65.92 ZIP. Trains nothing.
# Runs on a Blackwell slice because the original ZIP was made there (jobs 341089, 341679): the same GPU type gives
# bit-identical boxes (MATCH); another type gives CLOSE MATCH (job 343967 on an A40: boxes within 1 px).
#SBATCH --job-name=job-z
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_pro_6000_blackwell_max
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-z-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u ~/sbatch-staging/job_z.py --out runs/job-z-$SLURM_JOB_ID/final.zip \
  --detector runs/ft40-s1adfsr2-340754/train/weights/last.pt \
  --branch runs/soup-s1fs/fliprefit-heads-341248/branch_head.pt \
  --conv runs/soup-s1fs/fliprefit-heads-341248/conv_head.pt \
  --compare nonsubmissions/281026/alldata-s1fsr2/alldata-s1fsr2-rounded-hedged-s1two-tail.zip
echo "end $(date '+%m-%d %H:%M')"
