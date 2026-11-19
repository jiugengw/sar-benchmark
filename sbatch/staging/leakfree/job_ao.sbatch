#!/bin/bash
# Leak-free retrain: the TA's reference model (ViT-B/16 ImageNet-21k, frozen, LoRA r16 on qkv/proj, classifier +
# Faster R-CNN heads, 24 epochs, models/terramind/train.py as the TA ran it) on the new train split, seeds 0 1 2,
# then its predictions on the new test and val, scored like ours. The mean over the seeds is the new Δm reference.
# Output runs/leakfree/results/ta-ref-<job>/.
#SBATCH --job-name=job-ao
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
YOLO_DATA=0 source ~/sbatch-staging/leakfree/env.sh
O=results/ta-ref-$SLURM_JOB_ID; mkdir -p $O
for seed in ${SEEDS:-0 1 2}; do
  python -u models/terramind/train.py --data data/course-data --backbone vit --init pretrained --adapt lora --seed $seed \
    --out runs/ta-ref-$SLURM_JOB_ID-s$seed --results $O/train --checkpoints $O/ckpt
  RUN=$(dirname $(ls runs/ta-ref-$SLURM_JOB_ID-s$seed/*/model.pt | head -1))
  for s in test val; do
    python -u models/terramind/make_submission.py --run $RUN --data data/course-data --split $s --out $O/s$seed-$s.zip
    python scripts/score_released.py $O/s$seed-$s.zip $s > $O/score-s$seed-$s.json
    echo "seed $seed $s: $(tr -d '\n ' < $O/score-s$seed-$s.json)"
  done
done
echo LF_TA_DONE
