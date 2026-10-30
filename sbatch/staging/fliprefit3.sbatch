#!/bin/bash
# All-data refit 3 (user, 28 Oct ~17:15: "sure"): refit 2 (340754, 19 flip epochs on the flip soup, Codabench mAP 59.93
# with the quick classifier) plus 10 more epochs of the same class-selective flip memorise (lr 0.001, beta 8).
# The ladder so far: 0 epochs 59.80, 9 epochs 59.75, 19 epochs 59.93; this is the 29-epoch rung.
#SBATCH --job-name=sar-ft40-fliprefit3
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --time=08:00:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-fr3-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u models/yolo/finetune_detector.py --tag ft40-s1adfsr3-$SLURM_JOB_ID \
    --source runs/ft40-s1adfsr2-340754/train/weights/last.pt \
    --classifier runs/memorise-lr2-332326-package/image_classifier.pt --all-data --skip-evaluation --no-augment \
    --lr0 0.001 --epochs 10 --stem-stride 1 --batch 16 --tal-beta 8 --dataset-flips \
    --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine && echo FLIPREFIT3_ALLDATA_DONE
