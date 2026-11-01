#!/bin/bash
# job_v: the finer-stride check and ZIP (job_v.py).
#   V=check sbatch --job-name=job13 --export=ALL ~/sbatch-staging/job_v.sbatch   holdout: 344678 as is vs routed
#   V=pack VARIANT=six sbatch --job-name=job14 --export=ALL ~/sbatch-staging/job_v.sbatch
#           the +65.80 ZIP with the routed test images redone at the finer stride, then job_w's extras
#SBATCH --job-name=job-v
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
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-v-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 SAR_INPUT_FILTER=none
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader); variant ${V:?set V}"
J="python -u $HOME/sbatch-staging/job_v.py"
BASE=runs/ft40-s1r-flipsel-344678/train/weights/best.pt
SOUP=runs/soup-s1fs/alldata-s1fs-six.pt
REF=nonsubmissions/271026/alldata-s1fs-soup6/alldata-s1fs-soup6-rounded-hedged-s1two-tail.zip
D=results/v-$SLURM_JOB_ID
case $V in
  check)
    $J patch $BASE $D/twin-fine.pt
    $J export $BASE holdout $D/coarse.json
    $J export $D/twin-fine.pt holdout $D/fine.json
    $J verdict $D/coarse.json $D/fine.json 2>&1 | grep -v -i warn ;;
  pack)
    N=alldata-s1fs-soup6-s1two-tail-fine${VARIANT:?set VARIANT}
    $J patch $SOUP runs/soup-s1fs/alldata-s1fs-six-fine.pt
    $J export runs/soup-s1fs/alldata-s1fs-six-fine.pt test $D/fine.json
    $J export runs/soup-s1fs/alldata-s1fs-six-fine.pt test $D/fine95.json 0.95 0.001 1000
    $J pack $REF $D/fine.json $D/fine95.json results/w-348180/nms95.json $VARIANT submissions/$N/$N.zip $D/nms95.json
    python -u ~/sbatch-staging/job_w.py merge submissions/$N/$N.zip $D/nms95.json submissions/$N-extras/$N-extras.zip
    sha256sum submissions/$N/$N.zip submissions/$N-extras/$N-extras.zip ;;
  *) echo "unknown variant $V"; exit 2 ;;
esac
echo "JOB_V_DONE $V $(date '+%m-%d %H:%M')"
