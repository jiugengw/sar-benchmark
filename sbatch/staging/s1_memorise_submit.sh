#!/bin/bash
# waits for a free submit slot after the stride-1 plain stage ends, then queues its memorise after the third stage
cd ~/scratchDirectory/sar-benchmark
while squeue -j 335304 -h 2>/dev/null | grep -q .; do sleep 120; done
J4=$(sbatch --parsable --time=09:00:00 --dependency=afterok:335306 sbatch/finetune_detector.sbatch s1-memorise --source runs/ft40-s1-third-335306/train/weights/best.pt --classifier runs/third-stage-331789-package/image_classifier.pt --no-augment --lr0 0.002 --tal-beta 8 --stem-stride 1)
echo "memorise submitted: $J4 (after 335306)" >> ~/sbatch-staging/s1_chain.log
