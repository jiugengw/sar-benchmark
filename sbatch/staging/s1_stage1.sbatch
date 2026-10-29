#!/bin/bash
# Stage 1 of the best model's chain (docs/BEST_MODEL_RECIPE.md): YOLO26m-P2 from Ultralytics' DOTA weights, first
# convolution at stride 1, 100 epochs on the training split. Submit from the repository root. If it hits the 24 h
# limit, resume with: RESUME=runs/yolo26m-p2-dota-s1-<job>/train/weights/last.pt sbatch --export=ALL s1_stage1.sbatch
# (As it ran on 23 Oct, written out by reset_s1.sh.)
#SBATCH --job-name=sar-s1-100
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/s1-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
if [ -n "${RESUME:-}" ]; then
  python -u -c "from ultralytics import YOLO; YOLO('$RESUME').train(resume=True)"
else
  python -u models/yolo/train_yolo26s_p2.py --scale m --weights pretrained/yolo26m-obb.pt --train-output runs/yolo26m-p2-dota-s1-$SLURM_JOB_ID --epochs 100 --batch 16 --workers 4 --flipud 0.0 --close-mosaic 10 --stem-stride 1
fi
echo S1_STAGE1_DONE
