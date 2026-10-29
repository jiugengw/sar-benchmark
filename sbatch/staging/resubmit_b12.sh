#!/bin/bash
# The flip soup's s1adf-b12 ingredient (338834) was cancelled while pending to let the interim ZIP's packaging job in.
# Resubmit it, identically, once flipsoup2.sh has submitted its last ingredient (so the two never race for the
# 4-submitted-jobs limit) and a place is free. flipsoup2.sh then finds it by name like any other ingredient.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/flipsoup.log
until grep -q "s1adf-b10: job [0-9]" $LOG; do sleep 120; done
while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done
T=runs/ft40-s1ad-third-338033/train/weights/last.pt; CLS=runs/memorise-lr2-332326-package/image_classifier.pt
EP=$(cat ~/sbatch-staging/flipsoup_epochs 2>/dev/null || echo 20); TL=$(printf "%02d:00:00" $((EP * 3 / 10 + 3)))
J=$(sbatch --parsable --time=$TL --gres=gpu:l40s:1 sbatch/finetune_detector.sbatch s1adf-b12 --source $T --classifier $CLS --all-data --skip-evaluation --no-augment --lr0 0.002 --epochs $EP --stem-stride 1 --batch 16 --dataset-flips --tal-beta 12)
echo "$(date '+%m-%d %H:%M') s1adf-b12 (resubmitted): job ${J:-FAILED}" >> $LOG
