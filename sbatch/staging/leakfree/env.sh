# Sourced by every leak-free retrain job (README.md here): a private copy of the code in /tmp on the node, with the
# course data laid out by the leak-free split, so every training and packaging script runs unchanged (the scripts
# have data/course-data, data/yolo11s-sar-512 and two holdout split.json paths built in).
#   runs/, results/  ->  ~/scratchDirectory/sar-benchmark/runs/leakfree/{runs,results} (kept)
#   data/course-data  <- runs/leakfree/course-data.tar (job_aj);  data/yolo11s-sar-512 built here (YOLO_DATA=0 skips)
# WS_ID reuses an earlier job's /tmp path (an Ultralytics resume needs the same data path).
R=~/scratchDirectory/sar-benchmark; LF=$R/runs/leakfree; W=/tmp/lf-${WS_ID:-$SLURM_JOB_ID}
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source $R/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=$W/mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CLS=$R/runs/memorise-lr2-332326-package/image_classifier.pt   # only for the skipped evaluation, as in the recipe
rm -rf $W; mkdir -p $W/data $LF/runs $LF/results; trap 'rm -rf $W' EXIT
tar -C $R --exclude=__pycache__ -cf - models scripts sarbench pretrained | tar -C $W -xf -
ln -s $LF/runs $W/runs; ln -s $LF/results $W/results
tar -C $W/data -xf $LF/course-data.tar
cd $W; export PYTHONPATH=$W:$W/models/yolo
if [ "${YOLO_DATA:-1}" = 1 ]; then
  python models/yolo/prepare_dataset.py --data data/course-data --split results/yolo26m-100-p2-328339/seed_0/split.json \
    --out data/yolo11s-sar-512 > $W/prepare.log
fi
echo "workspace $W on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) $(date '+%m-%d %H:%M')"
