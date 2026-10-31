#!/bin/bash
# job_d: an all-data candidate from existing detectors, no detector training (31 Oct, Phase 2 generalisation goal).
# Averages the listed detectors (all from the all-data third stage ft40-s1ad-third-338033) into one model, trains its
# two-head image classifier (branch + conv head, as the final model's heads 339995), then runs make_final_zip.py's
# five steps (one2many boxes, rounding, hedging, low-score tail, two-head labels).
#   PARTS="runs/soup-s1/alldata-s1-six.pt:1 runs/soup-s1fs/alldata-s1fs-six.pt:1" NAME=d1 \
#     sbatch --job-name=job-d1 --export=ALL ~/sbatch-staging/job_d.sbatch
#SBATCH --job-name=job-d
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-d-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 SAR_INPUT_FILTER=none
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader); ${NAME:?set NAME}: ${PARTS:?set PARTS}"
W=runs/job-d-$NAME-$SLURM_JOB_ID
mkdir -p $W
python - $W/detector.pt $PARTS <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / 'sbatch-staging'))
from job_g import mix_weights
parts = [[p.rsplit(':', 1)[0], float(p.rsplit(':', 1)[1])] for p in sys.argv[2:]]
mix_weights(parts, Path(sys.argv[1]))
print('averaged', parts, flush=True)
PY
python -u models/yolo/branch_head_study.py --detector $W/detector.pt --output $W/branch_head.pt --epochs 20 --batch 16 --all-data
python -u models/yolo/conv_head_study.py --detector $W/detector.pt --output $W/conv_head.pt --batch 16 --epochs 30 --all-data
python -u models/yolo/make_final_zip.py --detector $W/detector.pt --branch $W/branch_head.pt --conv $W/conv_head.pt \
  --out $W/final.zip
echo "JOB_D_ZIP $W/final.zip $(date '+%m-%d %H:%M')"
