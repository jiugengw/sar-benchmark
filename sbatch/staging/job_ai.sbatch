#!/bin/bash
# job_ai: copy search over every labelled course image (train + val + test, released labels), for a leak-free re-split.
# Packs the images into one zip (tools/leakfree/pack.py), then runs the leakage audit's same-zoom search (pixsearch.py)
# and zoom search (scalesearch.py, all pairs) on the radar images (tools/leakfree/coursesearch.py).
# Files in ~/sbatch-staging/course/: pack.py coursesearch.py pixsearch.py scalesearch.py. STEPS="pix scale" by default.
#   sbatch --job-name=job84 ~/sbatch-staging/job_ai.sbatch
# Output: runs/course-search-<jobid>/{pix,scale}/, runs/course-search-<jobid>/course.zip, ids.csv.
#SBATCH --job-name=job-ai
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SEARCH_DIR=$HOME/sbatch-staging/course
OUT=${OUT:-runs/course-search-$SLURM_JOB_ID}; mkdir -p $OUT
echo "start $(date '+%m-%d %H:%M') on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
for f in pack coursesearch pixsearch scalesearch; do python -m py_compile $SEARCH_DIR/$f.py; done
[ -f $OUT/course.zip ] || python $SEARCH_DIR/pack.py data/course-data $OUT
for s in ${STEPS:-pix scale}; do
  echo "== $s $(date '+%H:%M')"
  python -u $SEARCH_DIR/coursesearch.py $s $OUT/course.zip $OUT/$s
done
echo "done $(date '+%m-%d %H:%M')"
