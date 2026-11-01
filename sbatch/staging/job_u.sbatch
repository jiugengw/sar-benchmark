#!/bin/bash
# job_u: rebuild the final ZIP with the six-step pipeline (make_submission.py --final; step 6, the extra boxes ranked
# last, added 1 Nov) and name it final.zip. Trains nothing. Compares the result with try 1 (the +65.80 ZIP plus job_w's
# extras: +65.85) and checks that steps 1-5 (<out>-work/tailed.zip) are byte for byte the +65.80 ZIP. Runs on a
# Blackwell slice like the original packaging (same GPU type = bit-identical boxes); job_w's NMS-0.95 pass ran on an
# L40S, so the step-6 boxes can only be a close match to try 1's.
#   sbatch --job-name=job15 ~/sbatch-staging/job_u.sbatch
#SBATCH --job-name=job-u
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_pro_6000_blackwell_max
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-u-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
OLD=nonsubmissions/271026/alldata-s1fs-soup6/alldata-s1fs-soup6-rounded-hedged-s1two-tail.zip
TRY1=submissions/alldata-s1fs-soup6-rounded-hedged-s1two-tail-extras/alldata-s1fs-soup6-rounded-hedged-s1two-tail-extras.zip
OUT=runs/job-u-$SLURM_JOB_ID/final.zip
python -u models/yolo/make_submission.py --final --out $OUT \
  --weights runs/soup-s1fs/alldata-s1fs-six.pt \
  --branch runs/soup-s1fs/heads-339995/branch_head.pt \
  --conv runs/soup-s1fs/heads-339995/conv_head.pt \
  --compare $TRY1
python - $OUT $OLD <<'EOF'
import sys, zipfile
from pathlib import Path
out, old = Path(sys.argv[1]), zipfile.ZipFile(sys.argv[2])
tailed = zipfile.ZipFile(out.with_name(out.stem + '-work') / 'tailed.zip')
print('steps 1-5 boxes byte-identical to the +65.80 ZIP:', tailed.read('detection.json') == old.read('detection.json'))
print('labels byte-identical to the +65.80 ZIP:', zipfile.ZipFile(out).read('classification.csv') == old.read('classification.csv'))
EOF
sha256sum $OUT
echo "JOB_U_DONE $(date '+%m-%d %H:%M')"
