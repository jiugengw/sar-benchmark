#!/bin/bash
# Leak-free label-transfer probe (NOT ALLOWED, a record only): probe_transfer.py on the final pipeline's new-test ZIP
# (IN, default runs/leakfree/results/final-373198/final-test.zip), then score_released.py on the new test labels.
# Output runs/leakfree/results/transfer-<job>/. Needs no GPU, but runs in the student partition like the rest.
#SBATCH --job-name=job-ap
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
YOLO_DATA=0 source ~/sbatch-staging/leakfree/env.sh
cp ~/sbatch-staging/transfer.py ~/sbatch-staging/transfer_full.py ~/sbatch-staging/leakfree/probe_transfer.py $W/
O=results/transfer-$SLURM_JOB_ID; IN=${IN:-results/final-373198/final-test.zip}
python -u probe_transfer.py $IN $O
for n in all near; do
  python scripts/score_released.py $O/transferfull-$n-not-allowed.zip test --per-class > $O/score-$n-test.json
  echo "transfer $n test: $(head -8 $O/score-$n-test.json | tr -d '\n ')"
done
echo LF_PROBE_DONE
