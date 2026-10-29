#!/bin/bash
# All-data refit 4 (goal Δm 70; user 28 Oct ~23:30: "after 3am need to run smth, make sure l40s is preferred"): refit 3
# (341403, 29 flip epochs) plus 10 more epochs of the same class-selective flip memorise (lr 0.001, beta 8), submitted
# by after3.sh only if the holdout rung 341702 gains (or the user's Codabench result says go).
#SBATCH --job-name=sar-ft40-fliprefit4
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
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-fr4-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u models/yolo/finetune_detector.py --tag ft40-s1adfsr4-$SLURM_JOB_ID \
    --source runs/ft40-s1adfsr3-341403/train/weights/last.pt \
    --classifier runs/memorise-lr2-332326-package/image_classifier.pt --all-data --skip-evaluation --no-augment \
    --lr0 0.001 --epochs 10 --stem-stride 1 --batch 16 --tal-beta 8 --dataset-flips \
    --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine && echo FLIPREFIT4_ALLDATA_DONE
