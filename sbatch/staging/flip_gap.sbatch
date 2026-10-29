#!/bin/bash
# Flipped-copy diagnosis for jgwong.2023's SAR model (29 Oct): runs from his project folder, writes only to the folder
# you submit from. About 10 minutes on one GPU. Submit from a folder holding flip_gap.py and this file:
#   sbatch flip_gap.sbatch
# Result: flip_gap.out in that folder (send it back).
#SBATCH --job-name=sar-flip-gap
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=flip_gap.slurm.out
set -uo pipefail
REPO=${REPO:-/common/home/users/j/jgwong.2023/scratchDirectory/sar-benchmark}
HERE=$SLURM_SUBMIT_DIR
cd "$REPO" || { echo "cannot enter $REPO (no read access?)" | tee "$HERE/flip_gap.out"; exit 1; }
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-fg-$SLURM_JOB_ID
export PYTHONPATH=$REPO:$REPO/models/yolo YOLO_CONFIG_DIR=/tmp/sar-fg-yolo-$SLURM_JOB_ID
{ echo "== $(date '+%m-%d %H:%M') flipped-copy diagnosis, 340514 (job $SLURM_JOB_ID on $(nvidia-smi --query-gpu=name --format=csv,noheader))"
  python -u "$HERE/flip_gap.py" runs/ft40-s1r-fliprefit-340514/train/weights/best.pt \
      results/ft40-s1r-fliprefit-340514/one2many "$HERE/flip_gap_work" 2>&1 | grep -v -i warn; } > "$HERE/flip_gap.out"
