#!/bin/bash
# Leak-free retrain: the TA's reference model (ViT-B/16 ImageNet-21k, frozen, LoRA r16 on qkv/proj, classifier +
# Faster R-CNN heads, 24 epochs, models/terramind/train.py as the TA ran it) on the new train split, seeds 0 1 2,
# then its predictions on the new test and val, scored like ours. The mean over the seeds is the new Δm reference.
# Output runs/leakfree/results/ta-ref-<job>/. Resume: TA_ID=<old job> sbatch --export=ALL job_ao.sbatch
#SBATCH --job-name=job-ao
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
YOLO_DATA=0 source ~/sbatch-staging/leakfree/env.sh
ID=${TA_ID:-$SLURM_JOB_ID}   # TA_ID=<earlier job>: continue it, reusing the seeds it already trained
O=results/ta-ref-$ID; mkdir -p $O
for seed in ${SEEDS:-0 1 2}; do
  ls runs/ta-ref-$ID-s$seed/*/model.pt >/dev/null 2>&1 || \
  python -u models/terramind/train.py --data data/course-data --backbone vit --init pretrained --adapt lora --seed $seed \
    --out runs/ta-ref-$ID-s$seed --results $O/train --checkpoints $O/ckpt
  RUN=$(dirname $(ls runs/ta-ref-$ID-s$seed/*/model.pt | head -1))
  for s in test val; do
    [ -f $O/s$seed-$s.zip ] || python -u models/terramind/make_submission.py --run $RUN --data data/course-data --split $s --out $O/s$seed-$s.zip
    python scripts/score_released.py $O/s$seed-$s.zip $s > $O/score-s$seed-$s.json
    echo "seed $seed $s: $(tr -d '\n ' < $O/score-s$seed-$s.json)"
  done
done
echo LF_TA_DONE
