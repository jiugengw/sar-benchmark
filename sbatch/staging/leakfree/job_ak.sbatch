#!/bin/bash
# Leak-free retrain, stage 1 (docs/BEST_MODEL_RECIPE.md): YOLO26m-P2 from the DOTA weights, stride-1 stem, 100 epochs
# on the new train split (holdout 939). Output runs/leakfree/runs/s1-<job>. If it hits 24 h, resume in the same
# workspace path: RESUME=1 WS_ID=<old job> OLD=<old job> sbatch --export=ALL job_ak.sbatch
#SBATCH --job-name=job-ak
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
if [ -n "${RESUME:-}" ]; then
  python -u -c "from ultralytics import YOLO; YOLO('runs/s1-$OLD/train/weights/last.pt').train(resume=True)"
else
  python -u models/yolo/train_yolo26s_p2.py --scale m --weights pretrained/yolo26m-obb.pt --train-output runs/s1-$SLURM_JOB_ID --epochs 100 --batch 16 --workers 4 --flipud 0.0 --close-mosaic 10 --stem-stride 1
fi
echo LF_STAGE1_DONE
