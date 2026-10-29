#!/bin/bash
#SBATCH --job-name=sar-rawcand
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/rawcand-$SLURM_JOB_ID
for R in ft40-m4rep-bp4-memorise-335138 ft40-m4repeat-beta8-333674 ft40-memorise-lr2-332326; do
  python -u models/yolo/raw_candidates.py --weights runs/$R/train/weights/best.pt --images results/$R/one2many/instances_val_holdout.json --split-root data/course-data/train --out results/$R/one2many/raw_candidates.json
done
