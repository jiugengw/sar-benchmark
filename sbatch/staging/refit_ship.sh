#!/bin/bash
# refit_ship.sh <job> <N>: when all-data refit N (job, run runs/ft40-s1adfsr<N>-<job>) ends, package it (rounded +
# hedged + low-score tail, fliprefit_package.sbatch), then the two-head classifier, then the tail on the two-head ZIP
# (CPU). Logs "alldata-s1fsr<N>: tail ZIP: <path>" for each ZIP to fliprefit.log.
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo
J=$1; N=$2; TAG=alldata-s1fsr$N; R=runs/ft40-s1adfsr$N-$J; LOG=~/sbatch-staging/fliprefit.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
log "refit_ship.sh started: job $J (all-data refit $N)"
st=$(wait_job $J)
grep -q "FLIPREFIT${N}_ALLDATA_DONE" runs/sar-ft40-fliprefit$N-$J.out && [ -f $R/train/weights/last.pt ] || { log "$TAG: refit ended $st without finished weights"; exit 1; }
DET=$R/train/weights/last.pt
slot; PKG=$(sbatch --parsable --export=ALL,TAG=$TAG,DET=$DET,RUN=$R ~/sbatch-staging/fliprefit_package.sbatch); log "$TAG: packaging $DET: $PKG"
st=$(wait_job $PKG)
Z=$(grep -o "S1FSR_ZIP_READY .*" runs/sar-s1fsr-pkg-$PKG.out | cut -d' ' -f2); T=$(grep -o "S1FSR_TAIL_READY .*" runs/sar-s1fsr-pkg-$PKG.out | cut -d' ' -f2)
[ -n "$Z" ] || { log "$TAG: packaging ended $st without a ZIP"; exit 1; }
log "$TAG: rounded + hedged ZIP: $Z"; log "$TAG: tail ZIP: ${T:-none}"
slot; HJ=$(sbatch --parsable --export=ALL,DET=$DET,Z=$Z ~/sbatch-staging/fliprefit_heads.sbatch); log "$TAG: two-head classifier: $HJ"
st=$(wait_job $HJ); Z2=$(grep -o 'S1FSR_TWOHEAD_ZIP [^ ]*' runs/sar-s1fsr-heads-$HJ.out | cut -d' ' -f2)
log "$TAG: two-head classifier ended: $st; ${Z2:-no ZIP}"
if [ -n "$Z2" ] && [ -n "$T" ] && python models/yolo/lowconf_tail.py $Z2 ${T%-rounded-hedged-tail.zip}-lowconf.zip ${Z2%.zip}-tail.zip >> $LOG 2>&1; then
  log "$TAG: tail ZIP: ${Z2%.zip}-tail.zip"; else log "$TAG: tail ZIP: none (two-head)"; fi
log "REFIT${N}_SHIP_DONE"
