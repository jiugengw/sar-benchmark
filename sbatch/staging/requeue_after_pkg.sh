#!/bin/bash
# After the single all-data stride-1 packaging job is submitted: requeue the soup variant pulled back to free a submit
# slot (s1ad-m4spk8, same options as alldata_s1_soup.sh), then restart the nearest-neighbour test driver.
cd ~/scratchDirectory/sar-benchmark
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
until grep -q "packaging:" ~/sbatch-staging/alldata_s1.log; do sleep 60; done
T=$(ls -d runs/ft40-s1ad-third-[0-9]* | head -1)/train/weights/last.pt
S="--source $T --classifier runs/memorise-lr2-332326-package/image_classifier.pt --all-data --skip-evaluation --no-augment --lr0 0.002 --epochs 40 --stem-stride 1 --batch 16"
M4="--speckle-classes airport,bridge,harbor,oil_tank,playground,wind_turbine --speckle-width 512 --speckle-looks 4"
slot; J=$(sbatch --parsable --time=05:00:00 --gres=gpu:l40s:1 sbatch/finetune_detector.sbatch s1ad-m4spk8 $S --tal-beta 8 --speckle-prob 0.8 $M4)
slot; J2=$(sbatch --parsable --time=05:00:00 --gres=gpu:l40s:1 sbatch/finetune_detector.sbatch s1ad-b10 $S --tal-beta 10); echo "$(date "+%m-%d %H:%M") s1ad-b10: job $J2 (requeued after packaging)" >> ~/sbatch-staging/alldata_s1_soup.log
echo "$(date "+%m-%d %H:%M") s1ad-m4spk8: job $J (requeued after packaging)" >> ~/sbatch-staging/alldata_s1_soup.log
nohup setsid ~/sbatch-staging/nn_run.sh > ~/sbatch-staging/nn_eval.out 2>&1 < /dev/null &
