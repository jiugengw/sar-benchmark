#!/bin/bash
# Leak-free retrain, stage 5: one flip-soup ingredient (20 epochs, no augmentation, dataset flips of six classes)
# from the stage-4 model SRC, with TAL beta BETA (the six: 8 12 10 7 9 11). Output runs/leakfree/runs/ft40-lf-soup-b<BETA>-<job>.
#SBATCH --job-name=job-am
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
source ~/sbatch-staging/leakfree/env.sh
python -u models/yolo/finetune_detector.py --tag ft40-lf-soup-b$BETA-$SLURM_JOB_ID --source $SRC --classifier $CLS --all-data --skip-evaluation --no-augment --lr0 0.002 --epochs 20 --batch 16 --stem-stride 1 --dataset-flips --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine --tal-beta $BETA
echo LF_SOUP_DONE
