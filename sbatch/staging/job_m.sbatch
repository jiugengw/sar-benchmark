#!/bin/bash
# job_m: check `models/yolo/make_submission.py --final` (uploaded as ~/sbatch-staging/job_m.py, a neutral name), which
# now holds the five steps of make_final_zip.py. Trains nothing.
#   1. Rebuild the final ZIP from the saved model and compare it with the uploaded one (+65.80): MATCH expected on a
#      Blackwell slice, the GPU type that made it (job 344390 gave MATCH there with make_final_zip.py).
#   2. The same on a separate folder of 60 Phase 1 images (up to 25 per image size, job_m_subset.py), the --images
#      path Phase 2 will use, compared with those images' part of the uploaded ZIP: MATCH, or CLOSE MATCH if other
#      batch neighbours move boxes by GPU rounding.
# Result (31 Oct, job 346225, 4 min): 1. MATCH, all 73,555 boxes and 1,426 labels. 2. CLOSE MATCH, the same 60 labels,
# 2,794 of 2,798 boxes within 1 px (median 0 px), 4 at the score cut-offs.
#SBATCH --job-name=job-m
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
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-m-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
OUT=runs/job-m-$SLURM_JOB_ID
REF=nonsubmissions/271026/alldata-s1fs-soup6/alldata-s1fs-soup6-rounded-hedged-s1two-tail.zip
W="--weights runs/soup-s1fs/alldata-s1fs-six.pt --branch runs/soup-s1fs/heads-339995/branch_head.pt --conv runs/soup-s1fs/heads-339995/conv_head.pt"
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "#### 1. Phase 1 images"
python -u ~/sbatch-staging/job_m.py --final --out $OUT/final.zip $W --compare $REF
echo "#### 2. Phase 1 images in a separate folder"
python -u ~/sbatch-staging/job_m_subset.py $REF $OUT/subset $OUT/subset-ref.zip
python -u ~/sbatch-staging/job_m.py --final --images $OUT/subset --out $OUT/subset.zip $W --compare $OUT/subset-ref.zip
echo "end $(date '+%m-%d %H:%M')"
