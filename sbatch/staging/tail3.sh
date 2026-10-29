#!/bin/bash
# Adds the low-score tail (tail_zip.sbatch) to refit 3's ZIPs as fliprefit3.sh writes them: the quick-classifier ZIP,
# then the two-head one. Refit 2's tail ZIP scored mAP 60.04 (+0.11), so refit 3 is compared with its tail too.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/fliprefit.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
wait_line() { while ! grep -q -E "$1" $LOG; do sleep 120; done; grep -E "$1" $LOG | tail -1; }
DET=runs/ft40-s1adfsr3-341403/train/weights/last.pt
tailzip() {
  local J T; J=$(sbatch --parsable --export=ALL,DET=$DET,Z=$1 ~/sbatch-staging/tail_zip.sbatch); log "alldata-s1fsr3: adding the tail to $1: job $J"
  while squeue -j $J -h 2>/dev/null | grep -q .; do sleep 60; done
  T=$(grep -o "TAIL_ZIP_READY .*" runs/sar-tail-zip-$J.out | cut -d' ' -f2)
  log "alldata-s1fsr3: tail ZIP: ${T:-none, job ended $(sacct -j $J -X -n -o State | tr -d ' ')}"
}
log "tail3.sh started: tails for refit 3's ZIPs"
Z=$(wait_line "alldata-s1fsr3: (rounded \+ hedged ZIP|packaging ended)|refit 3 ended" | grep -o "nonsubmissions/[^ ]*\.zip")
[ -n "$Z" ] && tailzip $Z
Z=$(wait_line "alldata-s1fsr3: two-head classifier ended|FLIPREFIT3_DONE" | grep -o "nonsubmissions/[^ ]*s1two\.zip" | tail -1)
[ -n "$Z" ] && tailzip $Z
log TAIL3_DONE
