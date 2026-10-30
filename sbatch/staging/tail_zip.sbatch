#!/bin/bash
# Adds the low-score tail to a finished ZIP (user, 28 Oct ~22:00, after the holdout tail test gained +0.09): the ZIP's
# own detector predicts the test images again with conf 1e-5, and models/yolo/lowconf_tail.py appends its sub-0.001
# boxes below every box in the ZIP. Env: DET (weights), Z (rounded + hedged [-s1two] ZIP); writes Z with -tail.
#SBATCH --job-name=sar-tail-zip
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-tz-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader): $DET -> $Z"
W=runs/tail-zip-$SLURM_JOB_ID; mkdir -p $W
python -u models/yolo/make_submission.py --weights $DET --detection-branch one2many --out $W/lowconf.zip --conf 1e-5 --max-det 1000
T=${Z%.zip}-tail.zip
python models/yolo/lowconf_tail.py $Z $W/lowconf.zip $T
echo "TAIL_ZIP_READY $T"
