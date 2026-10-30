#!/bin/bash
# job_z: rebuild the final ZIP from the saved model with models/yolo/make_final_zip.py (uploaded as
# ~/sbatch-staging/job_z.py, a neutral name) and compare it with the uploaded ZIP. Trains nothing. The final model
# (30 Oct) is the flip soup without refits: +65.80. (Refit 2, +65.92, rebuilt byte for byte in job 344078.)
# Runs on a Blackwell slice because the original ZIP was made there (jobs 339980, 339995, 344180): the same GPU type gives
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
  --detector runs/soup-s1fs/alldata-s1fs-six.pt \
  --branch runs/soup-s1fs/heads-339995/branch_head.pt \
  --conv runs/soup-s1fs/heads-339995/conv_head.pt \
  --compare nonsubmissions/271026/alldata-s1fs-soup6/alldata-s1fs-soup6-rounded-hedged-s1two-tail.zip
echo "end $(date '+%m-%d %H:%M')"
