#!/bin/bash
# Slot 2 while the flip-refit probe packages (28 Oct). Two parts in one job:
# 1. Holdout SWA, no training: average the flip test (338727) with its 10-epoch refit (340514) and score by holdout
#    group. The refit came from a warm restart (lr back up to 0.001), and the average of the two ends of such a cycle
#    often beats both. If it does, the all-data twin is free: average the +65.55 soup with the 9-epoch all-data refit.
# 2. All-data refit 2: the 9-epoch all-data refit (340515) plus 10 more epochs of the same class-selective flip
#    memorise (lr 0.001, beta 8). The 10-epoch holdout refit gained in every group but near-copies; this continues it.
#SBATCH --job-name=sar-ft40-fliprefit2
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|rtx_pro_6000_blackwell_max
#SBATCH --time=08:00:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-fr2-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
OUT=~/sbatch-staging/fliprefit_eval.out
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python - <<'PY'
from pathlib import Path
from swa_study import average
from models.yolo.deadline_recovery import evaluate_export
out = Path('results/swa-s1r-flip-refit'); out.mkdir(parents=True, exist_ok=True)
w = average([Path('runs/ft40-s1r-flipmem-338727/train/weights/best.pt'),
             Path('runs/ft40-s1r-fliprefit-340514/train/weights/best.pt')], out / 'averaged.pt')
evaluate_export(w, Path('runs/third-stage-331789-package/image_classifier.pt'), 'one2many', out / 'one2many')
print('SWA_SCORED', flush=True)
PY
[ -f results/swa-s1r-flip-refit/one2many/predictions_val.json ] && {
  echo "== $(date '+%m-%d %H:%M') holdout SWA: flip test + its refit (job $SLURM_JOB_ID)" >> $OUT
  python ~/sbatch-staging/group_runs.py ft40-s1r-flipmem-338727 ft40-s1r-fliprefit-340514 swa-s1r-flip-refit 2>&1 | grep -v -i warn >> $OUT; }
echo SWA_DONE >> $OUT
python -u models/yolo/finetune_detector.py --tag ft40-s1adfsr2-$SLURM_JOB_ID \
    --source runs/ft40-s1adfsr-soup6-340515/train/weights/last.pt \
    --classifier runs/memorise-lr2-332326-package/image_classifier.pt --all-data --skip-evaluation --no-augment \
    --lr0 0.001 --epochs 10 --stem-stride 1 --batch 16 --tal-beta 8 --dataset-flips \
    --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine && echo FLIPREFIT2_ALLDATA_DONE
