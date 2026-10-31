#!/bin/bash
# job_q: submit the remaining job_n variants one at a time whenever fewer than 4 of the user's jobs are submitted
# (the cluster's limit), so the queue stays full without going over it. Start detached on the login node:
#   setsid nohup bash ~/sbatch-staging/job_q.sh >> ~/sbatch-staging/job_q.log 2>&1 < /dev/null &
cd ~/scratchDirectory/sar-benchmark
k=3
for V in dedupe wd5e3; do
  k=$((k + 1))
  while [ "$(squeue -u $USER -h | wc -l)" -ge 4 ]; do sleep 300; done
  J=$(V=$V sbatch --parsable --job-name=job-n$k --export=ALL ~/sbatch-staging/job_n.sbatch) \
    && echo "$(date '+%m-%d %H:%M') job-n$k $V $J" || echo "$(date '+%m-%d %H:%M') job-n$k $V NOT SUBMITTED"
done
echo "$(date '+%m-%d %H:%M') JOB_Q_DONE"
