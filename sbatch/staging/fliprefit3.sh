#!/bin/bash
# Turns job fliprefit3.sbatch into probe ZIPs: packaged (rounded + hedged, quick classifier) then the stride-1
# two-head classifier, the same as refit 2 in fliprefit2.sh.
#   fliprefit3.sh <job>
cd ~/scratchDirectory/sar-benchmark
J=$1
LOG=~/sbatch-staging/fliprefit.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
ship() {  # TAG DET RUN: package, then the two-head classifier
  local TAG=$1 DET=$2 RUN=$3 PKG HJ st Z
  slot; PKG=$(sbatch --parsable --export=ALL,TAG=$TAG,DET=$DET,RUN=$RUN ~/sbatch-staging/fliprefit_package.sbatch)
  log "$TAG: packaging $DET: $PKG"
  st=$(wait_job $PKG)
  Z=$(grep -o "nonsubmissions/[^ ]*$TAG-rounded-hedged.zip" runs/sar-s1fsr-pkg-$PKG.out | tail -1)
  [ -n "$Z" ] || { log "$TAG: packaging ended $st without a ZIP"; return 1; }
  log "$TAG: rounded + hedged ZIP: $Z"
  slot; HJ=$(sbatch --parsable --export=ALL,DET=$DET,Z=$Z ~/sbatch-staging/fliprefit_heads.sbatch); log "$TAG: two-head classifier: $HJ"
  st=$(wait_job $HJ); log "$TAG: two-head classifier ended: $st; $(grep -o 'S1FSR_TWOHEAD_ZIP.*' runs/sar-s1fsr-heads-$HJ.out | tail -1)"
}
log "fliprefit3.sh started: job $J (all-data refit 3, 29 flip epochs)"
st=$(wait_job $J)
R3=runs/ft40-s1adfsr3-$J
if grep -q FLIPREFIT3_ALLDATA_DONE runs/sar-ft40-fliprefit3-$J.out && [ -f $R3/train/weights/last.pt ]; then
  ship alldata-s1fsr3 $R3/train/weights/last.pt $R3
else log "all-data refit 3 ended $st without finished weights"; fi
log FLIPREFIT3_DONE
