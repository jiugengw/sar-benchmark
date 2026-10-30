#!/bin/bash
# job_r_ship.sh <job A> <job B>: soup of refits. When both job_r jobs end, average refit 2 with every finished variant
# into one model (siblings of one flip soup), package it (job_p: quick classifier, rounding, hedging, low-score tail),
# train its two-head classifier (job_h) and add the tail to that ZIP. Neutral job names. Log: ~/sbatch-staging/job_r.log
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo
A=$1; B=$2; TAG=alldata-s1fsrsoup; LOG=~/sbatch-staging/job_r.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
log "job_r_ship.sh started: variant jobs $A $B"
sa=$(wait_job $A); sb=$(wait_job $B); log "variant jobs ended: $A $sa, $B $sb"
P=runs/ft40-s1adfsr2-340754/train/weights/last.pt
for j in $A $B; do
  for w in $(grep -h -o 'VARIANT_DONE b[0-9]* [^ ]*' runs/job-r*-$j.out 2>/dev/null | cut -d' ' -f3); do [ -f $w ] && P="$P $w"; done
done
n=$(echo $P | wc -w); log "averaging $n models: $P"
[ $n -ge 3 ] || { log "only $n models finished; no soup"; exit 1; }
mkdir -p runs/soup-s1fs; DET=runs/soup-s1fs/alldata-s1fsr-soup$n.pt
python - $DET $P <<'PY' >> $LOG 2>&1
import sys
from pathlib import Path
from swa_study import average
average([Path(p) for p in sys.argv[2:]], Path(sys.argv[1])); print('averaged', len(sys.argv) - 2)
PY
[ -f $DET ] || { log "averaging failed"; exit 1; }
slot; PKG=$(sbatch --parsable --job-name=job-p --export=ALL,TAG=$TAG,DET=$DET,RUN=runs/ft40-s1adfsr2-340754 ~/sbatch-staging/job_p.sbatch)
log "soup $DET; packaging: $PKG"
st=$(wait_job $PKG)
Z=$(grep -o "S1FSR_ZIP_READY .*" runs/job-p-$PKG.out | cut -d' ' -f2); T=$(grep -o "S1FSR_TAIL_READY .*" runs/job-p-$PKG.out | cut -d' ' -f2)
[ -n "$Z" ] || { log "packaging ended $st without a ZIP"; exit 1; }
log "quick ZIP, upload first (mAP decides; refit 2 had 60.04): ${T:-$Z}"
slot; HJ=$(sbatch --parsable --job-name=job-h --export=ALL,DET=$DET,Z=$Z ~/sbatch-staging/job_h.sbatch); log "heads: $HJ"
st=$(wait_job $HJ); Z2=$(grep -o 'S1FSR_TWOHEAD_ZIP [^ ]*' runs/job-h-$HJ.out | cut -d' ' -f2)
log "heads ended: $st; ${Z2:-no ZIP}"
if [ -n "$Z2" ] && [ -n "$T" ] && python models/yolo/lowconf_tail.py $Z2 ${T%-rounded-hedged-tail.zip}-lowconf.zip ${Z2%.zip}-tail.zip >> $LOG 2>&1; then
  log "final ZIP: ${Z2%.zip}-tail.zip"; else log "final ZIP: no tail added (two-head ZIP: ${Z2:-none})"; fi
log "JOB_R_SHIP_DONE"
