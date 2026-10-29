#!/bin/bash
# All-data side of the flip refit long shot (28 Oct): the +65.55 flip soup (runs/soup-s1fs/alldata-s1fs-six.pt) plus
# 10 more epochs of its class-selective flip memorise at lr 0.001 (beta 8). Started alongside the holdout test
# (fliprefit_test.sbatch); fliprefit.sh packages it only if that test passes.
#SBATCH --job-name=sar-ft40-fliprefit
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --time=12:00:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-fra-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u models/yolo/finetune_detector.py --tag ft40-s1adfsr-soup6-$SLURM_JOB_ID --source runs/soup-s1fs/alldata-s1fs-six.pt \
    --classifier runs/memorise-lr2-332326-package/image_classifier.pt --all-data --skip-evaluation --no-augment \
    --lr0 0.001 --epochs 10 --stem-stride 1 --batch 16 --tal-beta 8 --dataset-flips \
    --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine && echo FLIPREFIT_ALLDATA_DONE
