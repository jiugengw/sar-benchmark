#!/bin/bash
# job_y: job_x's verdict and ZIP (job_y.py). Start once the job_x runs have stopped (a last.pt being written can't be read).
#   V=check sbatch --job-name=job8 --export=ALL ~/sbatch-staging/job_y.sbatch   twin head on the holdout: GO or STOP
#   V=pack  sbatch --job-name=job9 --export=ALL --constraint=rtx_pro_6000_blackwell_max ~/sbatch-staging/job_y.sbatch
#           all-data head into the flip soup and the ZIP, on a Blackwell like the +65.80 packaging, so the image labels
#           come out byte for byte the same (--compare checks)
# TWIN / ALL override which job_x checkpoints are used.
#SBATCH --job-name=job-y
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-y-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 SAR_INPUT_FILTER=none
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader); variant ${V:?set V}"
Y="python -u $HOME/sbatch-staging/job_y.py"
TWIN=${TWIN:-runs/ft40-x-twin-348112/train/weights/last.pt}
ALL=${ALL:-runs/ft40-x-all-348113/train/weights/last.pt}
BASE=runs/ft40-s1r-flipsel-344678/train/weights/best.pt
SOUP=runs/soup-s1fs/alldata-s1fs-six.pt
REF=nonsubmissions/271026/alldata-s1fs-soup6/alldata-s1fs-soup6-rounded-hedged-s1two-tail.zip  # the uploaded +65.80 ZIP (job_t)
case $V in
  check)
    D=results/y-twin-$SLURM_JOB_ID
    $Y head $BASE $TWIN $D/twin-head.pt
    $Y export $BASE $D/base.json
    $Y export $D/twin-head.pt $D/new.json
    $Y verdict $D/base.json $D/new.json 2>&1 | grep -v -i warn ;;
  pack)
    N=alldata-s1fs-six-xhead-$SLURM_JOB_ID
    $Y head $SOUP $ALL runs/soup-s1fs/$N.pt
    python -u models/yolo/make_submission.py --final --weights runs/soup-s1fs/$N.pt \
      --branch runs/soup-s1fs/heads-339995/branch_head.pt --conv runs/soup-s1fs/heads-339995/conv_head.pt \
      --out submissions/$N/$N.zip --compare $REF ;;
  *) echo "unknown variant $V"; exit 2 ;;
esac
echo "JOB_Y_DONE $V $(date '+%m-%d %H:%M')"
