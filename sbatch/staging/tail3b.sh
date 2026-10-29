#!/bin/bash
# Refit 3's low-score tails without a GPU slot: the packaging job (fliprefit_package.sbatch) already wrote the quick
# ZIP's tail and kept lowconf.zip; the two-head ZIP has the same boxes, so its tail is added here on CPU.
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo
LOG=~/sbatch-staging/fliprefit.log; TAG=alldata-s1fsr3
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
wait_line() { while ! grep -q -E "$1" $LOG; do sleep 120; done; grep -E "$1" $LOG | tail -1; }
log "tail3b.sh started"
L=$(wait_line "$TAG: (rounded \+ hedged ZIP|packaging ended)|refit 3 ended")
PKG=$(grep -o "$TAG: packaging [^ ]*: [0-9]*" $LOG | tail -1 | grep -o "[0-9]*$")
T=$(grep -o "S1FSR_TAIL_READY .*" runs/sar-s1fsr-pkg-$PKG.out 2>/dev/null | cut -d' ' -f2)
log "$TAG: tail ZIP: ${T:-none (packaging job $PKG: $(grep -c S1FSR_TAIL_FAILED runs/sar-s1fsr-pkg-$PKG.out 2>/dev/null) failures)}"
[ -n "$T" ] || exit 1
LOW=${T%-rounded-hedged-tail.zip}-lowconf.zip
Z=$(wait_line "$TAG: two-head classifier ended|FLIPREFIT3_DONE" | grep -o "nonsubmissions/[^ ]*s1two\.zip" | tail -1)
if [ -n "$Z" ] && python models/yolo/lowconf_tail.py $Z $LOW ${Z%.zip}-tail.zip >> $LOG 2>&1; then log "$TAG: tail ZIP: ${Z%.zip}-tail.zip"
else log "$TAG: tail ZIP: none (two-head ZIP '${Z}')"; fi
log TAIL3B_DONE
