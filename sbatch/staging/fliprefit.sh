#!/bin/bash
# Flip-refit long shot (28 Oct), decided by the holdout test. Rule, fixed before the result: go if official-mix (clean
# holdout) gains >= +0.3 over the flip test 338727 and twins lose no more than 0.5. Go: wait for the all-data refit of
# the flip soup, package it (rounded + hedged), then the stride-1 two-head classifier. No: cancel the all-data refit.
#   fliprefit.sh <test job> <all-data job>
cd ~/scratchDirectory/sar-benchmark
TEST=$1; ALL=$2
LOG=~/sbatch-staging/fliprefit.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
log "fliprefit.sh started: holdout test $TEST, all-data refit $ALL"
wait_job $TEST > /dev/null
OUT=~/sbatch-staging/fliprefit_eval.out
F=$(grep "^ft40-s1r-flipmem-338727" $OUT | tail -1); R=$(grep "^ft40-s1r-fliprefit-$TEST" $OUT | tail -1)
if [ -z "$R" ]; then log "holdout test $TEST has no result row ($(sacct -j $TEST -X -n -o State | head -1 | tr -d ' ')): all-data refit $ALL cancelled"; scancel $ALL; exit 1; fi
read -r DM DT DF DN <<< $(python3 -c "
f = '$F'.split(); r = '$R'.split()
print(' '.join(f'{float(r[k]) - float(f[k]):+.2f}' for k in (1, 2, 3, 5)))")
log "flip test     $F"
log "flip refit    $R"
log "official-mix $DM, twins $DT, flipped copies $DF, new $DN"
if ! python3 -c "import sys; sys.exit(0 if $DM >= 0.3 and $DT >= -0.5 else 1)"; then
  scancel $ALL 2>/dev/null
  log "rule says NO (needs official-mix >= +0.3 and twins >= -0.5): all-data refit $ALL cancelled; +65.55 stands"; exit 0
fi
log "rule says GO: waiting for the all-data refit $ALL"
st=$(wait_job $ALL)
RUN=$(ls -d runs/ft40-s1adfsr-soup6-$ALL 2>/dev/null)
if [ -z "$RUN" ] || [ ! -f $RUN/train/weights/last.pt ] || ! grep -q FLIPREFIT_ALLDATA_DONE runs/sar-ft40-fliprefit-$ALL.out; then
  log "all-data refit $ALL ended $st without finished weights; no ZIP"; exit 1; fi
DET=$RUN/train/weights/last.pt
slot; PKG=$(sbatch --parsable --export=ALL,DET=$DET,RUN=$RUN ~/sbatch-staging/fliprefit_package.sbatch); log "all-data refit done ($DET); packaging: $PKG"
st=$(wait_job $PKG); log "packaging ended: $st"
Z=$(grep -o "nonsubmissions/[^ ]*alldata-s1fsr-rounded-hedged.zip" runs/sar-s1fsr-pkg-$PKG.out | tail -1)
[ -n "$Z" ] || { log "packaging did not finish; no ZIP"; exit 1; }
log "rounded + hedged ZIP: $Z"
slot; HJ=$(sbatch --parsable --export=ALL,DET=$DET,Z=$Z ~/sbatch-staging/fliprefit_heads.sbatch); log "two-head classifier: $HJ"
st=$(wait_job $HJ); log "two-head classifier ended: $st; $(grep -o 'S1FSR_TWOHEAD_ZIP.*' runs/sar-s1fsr-heads-$HJ.out | tail -1)"
log FLIPREFIT_DONE
