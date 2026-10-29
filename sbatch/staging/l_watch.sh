#!/bin/bash
# The l stride-1 100-epoch stage now runs on a full L40S (job 336693). When it ends, log completion so the SPD chain
# (spd_chain.sh, waiting on this line) starts; resume from last.pt on the same GPU type if the job hits its time limit.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/l_s1.log
J=336693; R1=runs/yolo26l-p2-dota-s1-336693
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
STATE=$(wait_job $J); echo "$(date '+%m-%d %H:%M') l 100-epoch stage ended: $STATE" >> $LOG
while [ "$STATE" = "TIMEOUT" ]; do
  J=$(RESUME=$R1/train/weights/last.pt sbatch --parsable --export=ALL /tmp/l_s1_l40s.sbatch); echo "$(date '+%m-%d %H:%M') resuming: job $J" >> $LOG
  STATE=$(wait_job $J)
done
[ -f $R1/train/weights/best.pt ] && echo "$(date '+%m-%d %H:%M') L_S1_STAGE1_COMPLETE $R1 (100-epoch stage only, L40S batch 8)" >> $LOG || echo "$(date '+%m-%d %H:%M') no best.pt, stopping" >> $LOG
