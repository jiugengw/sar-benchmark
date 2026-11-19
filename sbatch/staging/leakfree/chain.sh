#!/bin/bash
# Driver for the leak-free retrain, on the login node:  nohup bash ~/sbatch-staging/leakfree/chain.sh >/dev/null 2>&1 &
# Submits job_aj (data), job_ao (TA reference, Blackwell), job_ak (stage 1), job_al (stages 2-4), six job_am (soup),
# job_an (final ZIPs + scores), keeping at most 4 jobs submitted; Slurm runs 2 at once. Jobs are named job86, job87, ...
# Log: ~/sbatch-staging/leakfree/chain.log. Restart: SKIP_TA=1 N=<next free number> nohup bash chain.sh ... Stops at the first job that does not finish.
cd ~/scratchDirectory/sar-benchmark
S=~/sbatch-staging/leakfree; LOG=$S/chain.log; LF=$PWD/runs/leakfree; N=${N:-86}
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 300; done; }
sub() { slot; J=$(sbatch --parsable --job-name=job$N "$@"); log "job$N = $J: $*"; N=$((N + 1)); }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 600; done; }
need() { grep -q "$2" runs/job$3-$1.out 2>/dev/null || { log "job $1 did not finish ($2 missing), stopping"; exit 1; }; }
log "chain start"
if [ ! -f $LF/course-data.tar ]; then sub $S/job_aj.sbatch; P=$J; PN=$((N - 1)); wait_job $P; need $P LF_DATA_READY $PN; fi
[ -n "${SKIP_TA:-}" ] || { sub $S/job_ao.sbatch; TA=$J; }   # SKIP_TA=1: the TA job was submitted by hand
sub $S/job_ak.sbatch; S1=$J; S1N=$((N - 1)); wait_job $S1; need $S1 LF_STAGE1_DONE $S1N
sub --export=ALL,SRC=$LF/runs/s1-$S1/train/weights/best.pt $S/job_al.sbatch; S2=$J; S2N=$((N - 1))
wait_job $S2; need $S2 LF_STAGES234_DONE $S2N
THIRD=$LF/runs/ft40-lf-third-$S2/train/weights/last.pt; SOUP=""; IDS=""
for b in 8 12 10 7 9 11; do sub --export=ALL,SRC=$THIRD,BETA=$b $S/job_am.sbatch; IDS="$IDS $J:$((N - 1))"; SOUP="$SOUP $LF/runs/ft40-lf-soup-b$b-$J/train/weights/last.pt"; done
for x in $IDS; do wait_job ${x%%:*}; need ${x%%:*} LF_SOUP_DONE ${x#*:}; done
sub --export=ALL,SOUP="${SOUP# }" $S/job_an.sbatch; F=$J; FN=$((N - 1)); wait_job $F; need $F LF_FINAL_DONE $FN
grep -h "^final \|^plain " runs/job$FN-$F.out >> $LOG; grep -h "^seed " runs/job*-$TA.out >> $LOG 2>/dev/null
log "chain done"
