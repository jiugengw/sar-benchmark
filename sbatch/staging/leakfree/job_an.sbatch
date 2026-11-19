#!/bin/bash
# Leak-free retrain, stages 6-11: average the six soup ingredients (SOUP = their last.pt, space-separated), train
# the two-head classifier on all data, then make_submission.py --final on the new test and val images, and score
# with scripts/score_released.py against the new released labels. Also scores the plain detector pass (step 1
# boxes only, same image labels) to show what the packaging adds. Output runs/leakfree/results/final-<job>/.
#SBATCH --job-name=job-an
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
source ~/sbatch-staging/leakfree/env.sh
O=results/final-$SLURM_JOB_ID; mkdir -p $O; D=$O/soup.pt
python -c "import sys; from pathlib import Path; from swa_study import average; average([Path(p) for p in sys.argv[2:]], Path(sys.argv[1])); print('averaged', len(sys.argv) - 2)" $D $SOUP
python -u models/yolo/branch_head_study.py --detector $D --output $O/branch_head.pt --epochs 20 --batch 16 --all-data
python -u models/yolo/conv_head_study.py --detector $D --output $O/conv_head.pt --batch 16 --epochs 30 --all-data
for s in test val; do
  python -u models/yolo/make_submission.py --final --weights $D --branch $O/branch_head.pt --conv $O/conv_head.pt --images data/course-data/$s --out $O/final-$s.zip
  python - $O/final-$s-work/base.zip $O/final-$s.zip $O/plain-$s.zip <<'PY'
import sys, zipfile
base, final, out = sys.argv[1:]
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('detection.json', zipfile.ZipFile(base).read('detection.json'))
    z.writestr('classification.csv', zipfile.ZipFile(final).read('classification.csv'))
PY
  for k in final plain; do python scripts/score_released.py $O/$k-$s.zip $s --per-class > $O/score-$k-$s.json; echo "$k $s: $(head -8 $O/score-$k-$s.json | tr -d '\n ')"; done
done
echo LF_FINAL_DONE
