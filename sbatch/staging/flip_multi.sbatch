#!/bin/bash
# Several class-selective flip soup ingredients trained SIDE BY SIDE in one job, one GPU each (the account may run
# only 2 jobs at once, but a job may hold several GPUs). LIST="name:beta ..." from the environment. Batch 8: the
# Blackwell cards run out of memory at 16, and with Ultralytics' nominal batch 64 (gradient accumulation) batch 8
# trains almost exactly like 16.
#SBATCH --job-name=sar-ft40-multi
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=14:00:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
export PYTHONPATH=$PWD:$PWD/models/yolo
T=runs/ft40-s1ad-third-338033/train/weights/last.pt; CLS=runs/memorise-lr2-332326-package/image_classifier.pt
S="--source $T --classifier $CLS --all-data --skip-evaluation --no-augment --lr0 0.002 --epochs 20 --stem-stride 1 --batch 8 --dataset-flips --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine"
GPUS=(${CUDA_VISIBLE_DEVICES//,/ })
echo "GPUs given: ${CUDA_VISIBLE_DEVICES}"
i=0
for spec in $LIST; do
  name=${spec%%:*}; b=${spec#*:}
  echo "MULTI start $name (beta $b) on ${GPUS[$i]} $(date '+%m-%d %H:%M')"
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} MPLCONFIGDIR=/tmp/sar-multi-$SLURM_JOB_ID-$i \
    python -u models/yolo/finetune_detector.py --tag "ft40-${name}-${SLURM_JOB_ID}" $S --tal-beta $b \
    > runs/sar-ft40-multi-${SLURM_JOB_ID}-${name}.out 2>&1 &
  i=$((i + 1)); sleep 60
done
wait
echo MULTI_DONE
