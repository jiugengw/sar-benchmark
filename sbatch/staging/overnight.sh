#!/bin/bash
# Runs on the cluster login node, independent of the Mac: release relabelling once packaging starts,
# then resubmit the Objects365 pipeline once packaging has finished.
cd ~/scratchDirectory/sar-benchmark
log=~/sbatch-staging/overnight.log
until squeue -j 333681 -h -o %T 2>/dev/null | grep -q -v PENDING || ! squeue -j 333681 -h | grep -q .; do sleep 60; done
scontrol release 333581; echo "$(date +%H:%M) released relabel 333581" >> $log
while squeue -j 333681 -h 2>/dev/null | grep -q .; do sleep 30; done
echo "$(date +%H:%M) packaging done: $(grep -E "CANDIDATE" runs/sar-package-333681.out | tail -1)" >> $log
until [ $(squeue -u $USER -h | wc -l) -lt 4 ]; do sleep 60; done
THEN_MEMORISE=1 sbatch sbatch/yolo26_pretrained_full.sbatch obj365 m >> $log 2>&1
echo "$(date +%H:%M) resubmitted Objects365" >> $log
