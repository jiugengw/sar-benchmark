#!/bin/bash
# Stride 1 + SPD at backbone shrink layers 1,3,5 (exact copies), from the m stride-1 speckle model, same third-stage
# recipe as s1r-third (337119). Submitted as soon as a submit slot is free (4 per user).
cd ~/scratchDirectory/sar-benchmark
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
slot
J=$(sbatch --parsable --time=16:00:00 --gres=gpu:l40s:1 sbatch/finetune_detector.sbatch s1spd-third --source runs/ft40-s1r-speckle-336990/train/weights/best.pt --classifier runs/yolo26m-p2-dota-verified-330705/image_classifier.pt --lr0 0.001 --stem-stride 1 --spd-layers 1,3,5 --repeat-classes airport,harbor,oil_tank,wind_turbine,bridge:4,playground:4 --repeat-factor 2 --repeat-width 512 --skip-evaluation --batch 16)
echo "$(date "+%m-%d %H:%M") stride-1 + SPD layers 1,3,5 third (control s1r-third 337119): job $J (runs/ft40-s1spd-third-$J)" >> ~/sbatch-staging/p1.log
