#!/bin/bash
# Turns job 340754 (fliprefit2.sbatch) into probe ZIPs, each packaged (rounded + hedged) then given the stride-1
# two-head classifier:
# - SWA ZIP: the +65.55 soup averaged with the 9-epoch all-data refit, built only if the holdout SWA (flip test +
#   its refit) beats the refit alone by >= +0.15 official-mix with twins no more than 0.5 lower (rule fixed before).
# - Refit-2 ZIP: the all-data refit continued 10 more epochs, always packaged (a probe ladder after the 9-epoch one).
#   fliprefit2.sh <job>
cd ~/scratchDirectory/sar-benchmark
J=$1
LOG=~/sbatch-staging/fliprefit.log; OUT=~/sbatch-staging/fliprefit_eval.out
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=8
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
log "fliprefit2.sh started: job $J (holdout SWA, then all-data refit 2)"
while ! grep -q SWA_DONE $OUT && squeue -j $J -h 2>/dev/null | grep -q .; do sleep 120; done
R=$(grep "^ft40-s1r-fliprefit-340514" $OUT | tail -1); W=$(grep "^swa-s1r-flip-refit" $OUT | tail -1)
if [ -n "$W" ]; then
  log "holdout refit  $R"; log "holdout SWA    $W"
  read -r DM DT <<< $(python3 -c "
r = '$R'.split(); w = '$W'.split(); print(f'{float(w[1]) - float(r[1]):+.2f} {float(w[2]) - float(r[2]):+.2f}')")
  log "SWA vs refit: official-mix $DM, twins $DT"
  if python3 -c "import sys; sys.exit(0 if $DM >= 0.15 and $DT >= -0.5 else 1)"; then
    python - <<'PY'
from pathlib import Path
from swa_study import average
average([Path('runs/soup-s1fs/alldata-s1fs-six.pt'), Path('runs/ft40-s1adfsr-soup6-340515/train/weights/last.pt')],
        Path('runs/soup-s1fs/alldata-s1fswa.pt'))
PY
    log "SWA rule GO: runs/soup-s1fs/alldata-s1fswa.pt (soup + 9-epoch refit)"
    ship alldata-s1fswa runs/soup-s1fs/alldata-s1fswa.pt runs/ft40-s1adfsr-soup6-340515 &
  else log "SWA rule NO (needs >= +0.15 official-mix over the refit, twins >= -0.5)"; fi
else log "holdout SWA has no result row"; fi
st=$(wait_job $J)
R2=runs/ft40-s1adfsr2-$J
if grep -q FLIPREFIT2_ALLDATA_DONE runs/sar-ft40-fliprefit2-$J.out && [ -f $R2/train/weights/last.pt ]; then
  ship alldata-s1fsr2 $R2/train/weights/last.pt $R2 &
else log "all-data refit 2 ended $st without finished weights"; fi
wait
log FLIPREFIT2_DONE
