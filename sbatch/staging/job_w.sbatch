#!/bin/bash
# job_w: a finished ZIP plus the detector's own NMS-removed boxes and wider hedge copies, all ranked last (job_w.py).
# W = the detector that made R; R = the ZIP to extend; N = the new submission's name. Defaults: the +65.80 ZIP.
#   sbatch --job-name=job10 --export=ALL ~/sbatch-staging/job_w.sbatch
#SBATCH --job-name=job-w
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
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-w-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 SAR_INPUT_FILTER=none
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
W=${W:-runs/soup-s1fs/alldata-s1fs-six.pt}
R=${R:-nonsubmissions/271026/alldata-s1fs-soup6/alldata-s1fs-soup6-rounded-hedged-s1two-tail.zip}
N=${N:-alldata-s1fs-soup6-rounded-hedged-s1two-tail-extras}
D=results/w-$SLURM_JOB_ID
python -u ~/sbatch-staging/job_w.py boxes $W $D/nms95.json
python -u ~/sbatch-staging/job_w.py merge $R $D/nms95.json submissions/$N/$N.zip
sha256sum submissions/$N/$N.zip
echo "JOB_W_DONE $(date '+%m-%d %H:%M')"
