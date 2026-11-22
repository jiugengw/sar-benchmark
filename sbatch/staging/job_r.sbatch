#!/bin/bash
# job_r: soup of refits (30 Oct). Siblings of refit 2 from the same flip
# soup, each with refit 2's recipe (two 10-epoch cycles of class-selective flip memorise at lr 0.001; refit 2 used
# beta 8) but another assigner beta. BETAS from the environment, run back to back (~10.5 h each):
#   BETAS="7 9" sbatch --job-name=job-r1 --export=ALL ~/sbatch-staging/job_r.sbatch
# job_r_ship.sh averages the finished variants with refit 2 into one model and packages it.
#SBATCH --job-name=job-r
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/jr-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader); betas ${BETAS:?set BETAS}"
SOUP=runs/soup-s1fs/alldata-s1fs-six.pt
COMMON="--classifier runs/memorise-lr2-332326-package/image_classifier.pt --all-data --skip-evaluation --no-augment
  --lr0 0.001 --epochs 10 --stem-stride 1 --batch 16 --dataset-flips
  --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine"
for B in $BETAS; do
  T1=ft40-s1adfsrv-b$B-c1-$SLURM_JOB_ID; T2=ft40-s1adfsrv-b$B-c2-$SLURM_JOB_ID
  python -u models/yolo/finetune_detector.py --tag $T1 --source $SOUP $COMMON --tal-beta $B \
    || { echo "VARIANT_FAILED b$B cycle 1"; continue; }
  python -u models/yolo/finetune_detector.py --tag $T2 --source runs/$T1/train/weights/last.pt $COMMON --tal-beta $B \
    || { echo "VARIANT_FAILED b$B cycle 2"; continue; }
  echo "VARIANT_DONE b$B runs/$T2/train/weights/last.pt $(date '+%m-%d %H:%M')"
done
echo JOB_R_DONE
