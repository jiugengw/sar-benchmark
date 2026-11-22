#!/bin/bash
# 28 Oct: do weight-averaged (SWA) classifier heads beat the last epoch? Training-split
# heads on 340514 (holdout twin of refit 2): branch 20 epochs (SWA 11-20), conv 30 epochs (SWA 16-30); two-head F1.
#SBATCH --job-name=sar-heads-swa-test
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --time=08:00:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/hswa-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
DET=runs/ft40-s1r-fliprefit-340514/train/weights/best.pt; H=runs/heads-swa-test-$SLURM_JOB_ID; OUT=~/sbatch-staging/heads_swa.out
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u models/yolo/branch_head_study.py --detector $DET --output $H/branch_head.pt --epochs 20 --batch 16 --swa-from 11
python -u models/yolo/conv_head_study.py --detector $DET --output $H/conv_head.pt --batch 16 --epochs 30 --swa-from 16
{ echo "== $(date '+%m-%d %H:%M') heads SWA test on 340514 (job $SLURM_JOB_ID)"; grep -h -E "^SWA|cross-fitted" runs/sar-heads-swa-test-$SLURM_JOB_ID.out
  python ~/sbatch-staging/heads_combo.py $H 2>&1 | grep -v -i warn; echo HEADS_SWA_DONE; } >> $OUT
