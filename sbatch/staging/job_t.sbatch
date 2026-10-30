#!/bin/bash
# job_t: add the low-score tail to the 27 Oct flip-soup ZIP (+65.55: everything but the refits and the tail), to
# measure the final recipe without refits. The flip soup's low-score boxes were never saved, so this predicts them
# (make_submission.py --conf 1e-5 --max-det 1000) and appends them with lowconf_tail.py, as for the +65.92 ZIP.
# Runs on a Blackwell slice, the GPU type that made the ZIP (package job 339980): lowconf_tail.py checks that the
# new run's boxes >= 0.001 equal the ZIP's, and another GPU type's rounding would fail that check.
#SBATCH --job-name=job-t
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_pro_6000_blackwell_max
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-t-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export SAR_INPUT_FILTER=none
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
D=nonsubmissions/271026/alldata-s1fs-soup6
W=runs/job-t-$SLURM_JOB_ID
mkdir -p $W
python -u models/yolo/make_submission.py --weights runs/soup-s1fs/alldata-s1fs-six.pt --detection-branch one2many \
  --out $W/lowconf.zip --conf 1e-5 --max-det 1000
python models/yolo/lowconf_tail.py $D/alldata-s1fs-soup6-rounded-hedged-s1two.zip $W/lowconf.zip \
  $D/alldata-s1fs-soup6-rounded-hedged-s1two-tail.zip
echo "TAIL_ZIP $D/alldata-s1fs-soup6-rounded-hedged-s1two-tail.zip"
echo "end $(date '+%m-%d %H:%M')"
