#!/bin/bash
# After the all-data soup's two-head ZIP is done, run a second seed of the stride-1 100-epoch stage (verification).
cd ~/scratchDirectory/sar-benchmark
until ls nonsubmissions/*/alldata-soup6/*-2head.zip >/dev/null 2>&1 || grep -q "Traceback" runs/sar-2head-*.out 2>/dev/null; do sleep 300; done
sed -e 's/--stem-stride 1$/--stem-stride 1 --seed 2/' -e 's/yolo26m-p2-dota-s1-/yolo26m-p2-dota-s1seed2-/' -e 's/sar-s1-100/sar-s1-seed2/' /tmp/s1_stage1.sbatch > /tmp/s1_seed2.sbatch
while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done
echo "$(date '+%m-%d %H:%M') stride-1 seed 2 (100 epochs): $(sbatch --parsable /tmp/s1_seed2.sbatch)" >> ~/sbatch-staging/reset_s1.log
