#!/bin/bash
# job_p2 (2 Nov): the Phase 2 ZIP. Runs the final pipeline (make_submission.py --final, six steps, the +65.85 model,
# training nothing) on the Phase 2 test images in data/course-data/test (2,065 images, released 2 Nov 00:01 SGT).
#   sbatch --job-name=job2 --constraint=<gpu> ~/sbatch-staging/job_p2.sbatch
#SBATCH --job-name=job-p2
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-p2-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
OUT=runs/job-p2-$SLURM_JOB_ID/phase2.zip
python -u models/yolo/make_submission.py --final --images data/course-data/test --out $OUT \
  --weights runs/soup-s1fs/alldata-s1fs-six.pt \
  --branch runs/soup-s1fs/heads-339995/branch_head.pt \
  --conv runs/soup-s1fs/heads-339995/conv_head.pt
python - $OUT data/course-data/test/images.json <<'PY'
import csv, io, json, sys, zipfile
from collections import Counter
z = zipfile.ZipFile(sys.argv[1]); ids = {im['id'] for im in json.load(open(sys.argv[2]))['images']}
dets = json.loads(z.read('detection.json')); rows = list(csv.DictReader(io.StringIO(z.read('classification.csv').decode())))
print('labels:', len(rows), 'all test ids:', {int(r[next(iter(r))]) for r in rows} == ids)
print('label counts:', sorted(Counter(r[list(r)[-1]] for r in rows).items()))
print('boxes:', len(dets), '>= 0.001:', sum(d['score'] >= 0.001 for d in dets), 'images with a box:', len({d['image_id'] for d in dets}))
PY
sha256sum $OUT
echo "JOB_P2_DONE $(date '+%m-%d %H:%M')"
