#!/bin/bash
# Leak-free retrain, stages 2-4 (all data = new train + holdout, last epoch kept), each from the one before:
# plain (lr 0.002) -> speckle (lr 0.001, speckle 0.5) -> third (lr 0.001, class repeats). SRC = stage 1 best.pt.
# Output runs/leakfree/runs/ft40-lf-{plain,speckle,third}-<job>.
#SBATCH --job-name=job-al
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
J=$SLURM_JOB_ID; C="--classifier $CLS --all-data --skip-evaluation --batch 16 --stem-stride 1"
ft() { [ -f runs/ft40-lf-$1-$J/train/weights/last.pt ] || python -u models/yolo/finetune_detector.py --tag ft40-lf-$1-$J "${@:2}"; }
ft plain --source $SRC $C --lr0 0.002
ft speckle --source runs/ft40-lf-plain-$J/train/weights/last.pt $C --lr0 0.001 --speckle-prob 0.5
ft third --source runs/ft40-lf-speckle-$J/train/weights/last.pt $C --lr0 0.001 --repeat-classes airport,harbor,oil_tank,wind_turbine,bridge:4,playground:4 --repeat-factor 2 --repeat-width 512
echo LF_STAGES234_DONE
