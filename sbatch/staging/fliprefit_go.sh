#!/bin/bash
# Probe ZIP from the flip refit (user, 28 Oct ~09:00: "package the 9-epoch all-data refit ... as a probe upload"),
# although the holdout rule said no (official-mix +0.14 < +0.3; twins +0.50, flipped copies +0.60, new +0.30).
# The all-data refit 340515 was cancelled during epoch 10; last.pt holds epoch 9.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/fliprefit.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
RUN=runs/ft40-s1adfsr-soup6-340515; DET=$RUN/train/weights/last.pt
slot; PKG=$(sbatch --parsable --export=ALL,DET=$DET,RUN=$RUN ~/sbatch-staging/fliprefit_package.sbatch)
log "user: package the 9-epoch refit anyway as a probe ($DET); packaging: $PKG"
st=$(wait_job $PKG); log "packaging ended: $st"
Z=$(grep -o "nonsubmissions/[^ ]*alldata-s1fsr-rounded-hedged.zip" runs/sar-s1fsr-pkg-$PKG.out | tail -1)
[ -n "$Z" ] || { log "packaging did not finish; no ZIP"; exit 1; }
log "rounded + hedged ZIP: $Z"
slot; HJ=$(sbatch --parsable --export=ALL,DET=$DET,Z=$Z ~/sbatch-staging/fliprefit_heads.sbatch); log "two-head classifier: $HJ"
st=$(wait_job $HJ); log "two-head classifier ended: $st; $(grep -o 'S1FSR_TWOHEAD_ZIP.*' runs/sar-s1fsr-heads-$HJ.out | tail -1)"
log FLIPREFIT_DONE
