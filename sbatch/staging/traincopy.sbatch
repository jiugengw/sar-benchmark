#!/bin/bash
# Memorise precision vs twin recognition (analysis, ~10 min): the flip test's AP on the training copies of the
# holdout twins vs on the twins themselves. Decides between the jitter test and a stronger memorise.
#SBATCH --job-name=sar-traincopy
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --time=01:00:00
#SBATCH --output=runs/sar-traincopy-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-traincopy-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
OUT=~/sbatch-staging/spktwin_eval.out
[ -f results/twin-copies.json ] || python ~/sbatch-staging/twin_label_agreement.py results/yolo26m-100-p2-328339/seed_0/instances_val_holdout.json results/twin-copies.json
echo "== $(date '+%m-%d %H:%M') flip test 338727 on the twins' training copies vs the twins (rounded + hedged)" >> $OUT
python ~/sbatch-staging/train_copy_ap.py results/twin-copies.json runs/ft40-s1r-flipmem-338727/train/weights/best.pt \
    runs/third-stage-331789-package/image_classifier.pt results/ft40-s1r-flipmem-338727/one2many 2>&1 | grep -v -i warn >> $OUT
echo TRAINCOPY_DONE >> $OUT
