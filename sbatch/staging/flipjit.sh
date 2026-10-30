#!/bin/bash
# Jitter soup, decided by the twin test (job 339893: the flip test's recipe plus brightness + sharpness jitter, holdout
# split). Rule, fixed before the result: twins are about a third of the test and official-mix (clean holdout) proxies
# the rest, so the expected test gain is E = 0.33 x (twins gain) + 0.67 x (official-mix gain) over the flip test
# (338727). Go if E >= +0.3 and twins gain >= +1.0. Then: four jitter ingredients in two chain jobs (b8 -> b10 and
# b9 -> b12), averaged into ONE model, packaged, rounded + hedged, and the stride-1 two-head classifier on it.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/flipjit.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=8
log "flipjit.sh started: waiting for the twin test 339893"
wait_job 339893 > /dev/null
OUT=~/sbatch-staging/spktwin_eval.out
row() { grep "^$1" $OUT | tail -1; }
F=$(row ft40-s1r-flipmem-338727); J=$(row ft40-s1r-flipjit-339893)
if [ -z "$J" ]; then log "twin test 339893 has no result row ($(sacct -j 339893 -X -n -o State | head -1 | tr -d ' ')): no jitter soup"; exit 1; fi
read -r DT DM E <<< $(python3 -c "
f = '$F'.split(); j = '$J'.split()
dm = float(j[1]) - float(f[1]); dt = float(j[2]) - float(f[2])
print(f'{dt:+.2f} {dm:+.2f} {0.33 * dt + 0.67 * dm:+.2f}')")
log "twin test: flip test  $F"
log "twin test: with jitter $J"
log "twins $DT, official-mix $DM, expected test gain E = $E"
if ! python3 -c "import sys; sys.exit(0 if $E >= 0.3 and $DT >= 1.0 else 1)"; then
  log "rule says NO (needs E >= +0.3 and twins >= +1.0): no jitter soup; +65.55 stands"; exit 0
fi
log "rule says GO: jitter soup of four (b8, b9, b10, b12)"
CHAIN=~/sbatch-staging/flipjit_chain.sbatch
A=""; B=""
until [ -n "$A" ]; do slot; A=$(LIST="s1adfj-b8:8 s1adfj-b10:10" sbatch --parsable --export=ALL $CHAIN); [ -z "$A" ] && sleep 60; done
until [ -n "$B" ]; do slot; B=$(LIST="s1adfj-b9:9 s1adfj-b12:12" sbatch --parsable --export=ALL $CHAIN); [ -z "$B" ] && sleep 60; done
log "chain jobs: $A (b8 -> b10), $B (b9 -> b12); any L40S / A40 / Blackwell card"
names="s1adfj-b8 s1adfj-b9 s1adfj-b10 s1adfj-b12"
find_run() {  # the finished run of this ingredient if there is one, else the newest
  for d in $(ls -d runs/ft40-$1-[0-9]* 2>/dev/null); do [ -f $d/compute.json ] && { echo $d; return; }; done
  ls -d runs/ft40-$1-[0-9]* 2>/dev/null | tail -1
}
JOBS="$A,$B"; resub=0
while :; do
  n=0; missing=""
  for x in $names; do r=$(find_run $x); if [ -n "$r" ] && [ -f $r/compute.json ]; then n=$((n+1)); else missing="$missing $x"; fi; done
  [ $n -eq 4 ] && break
  if ! squeue -j $JOBS -h 2>/dev/null | grep -q .; then
    [ $resub -eq 1 ] && { log "still missing after resubmitting:$missing; stopped"; exit 1; }
    resub=1; JOBS=""
    for x in $missing; do
      J=""; until [ -n "$J" ]; do slot; J=$(LIST="$x:${x##*-b}" sbatch --parsable --export=ALL --time=14:00:00 $CHAIN); [ -z "$J" ] && sleep 60; done
      JOBS="$JOBS${JOBS:+,}$J"; log "$x missing after the chains: resubmitted alone as job $J"
    done
  fi
  sleep 300
done
P=""; for x in $names; do P="$P $(find_run $x)/train/weights/last.pt"; done
log "averaging:$P"; mkdir -p runs/soup-s1fj
python - $P <<PY
import sys
from pathlib import Path
from swa_study import average
average([Path(p) for p in sys.argv[1:]], Path('runs/soup-s1fj/alldata-s1fj-four.pt')); print('averaged', len(sys.argv) - 1)
PY
sed -e 's/sar-s1soup-pkg/sar-s1fjsoup-pkg/' -e 's#runs/soup-s1/alldata-s1-six.pt#runs/soup-s1fj/alldata-s1fj-four.pt#' \
    -e 's/alldata-s1-soup6/alldata-s1fj-soup4/g' -e "s#runs/ft40-s1ad-memorise-[0-9]*#$(find_run s1adfj-b8)#" \
    -e 's/S1_SOUP_ZIP_READY/S1FJ_SOUP_ZIP_READY/' ~/sbatch-staging/s1soup_package.template.sbatch > ~/sbatch-staging/s1fjsoup_package.sbatch
grep -q "alldata-s1fj-four.pt" ~/sbatch-staging/s1fjsoup_package.sbatch || { log "package script rewrite failed"; exit 1; }
slot; PKG=$(sbatch --parsable --output=runs/%x-%j.out ~/sbatch-staging/s1fjsoup_package.sbatch); log "packaging: $PKG"
st=$(wait_job $PKG); log "packaging ended: $st"
grep -q S1FJ_SOUP_ZIP_READY runs/sar-s1fjsoup-pkg-$PKG.out || { log "packaging did not finish; no two-head ZIP"; exit 1; }
Z=$(grep -o "nonsubmissions/[^ ]*alldata-s1fj-soup4-rounded-hedged.zip" runs/sar-s1fjsoup-pkg-$PKG.out | tail -1)
log "rounded + hedged ZIP: $Z"
sed -e 's/sar-s1fssoup-heads/sar-s1fjsoup-heads/' -e 's#runs/soup-s1fs/alldata-s1fs-six.pt#runs/soup-s1fj/alldata-s1fj-four.pt#' \
    -e 's#H=runs/soup-s1fs/heads-$SLURM_JOB_ID#H=runs/soup-s1fj/heads-$SLURM_JOB_ID#' \
    -e "s#nonsubmissions/271026/alldata-s1fs-soup6/alldata-s1fs-soup6-rounded-hedged.zip#$Z#g" \
    -e 's/S1FS_SOUP_TWOHEAD_ZIP/S1FJ_SOUP_TWOHEAD_ZIP/' ~/sbatch-staging/s1fssoup_heads.sbatch > ~/sbatch-staging/s1fjsoup_heads.sbatch
grep -q "alldata-s1fj-four.pt" ~/sbatch-staging/s1fjsoup_heads.sbatch && grep -q "$Z" ~/sbatch-staging/s1fjsoup_heads.sbatch || { log "heads script rewrite failed"; exit 1; }
slot; HJ=$(sbatch --parsable --output=runs/%x-%j.out ~/sbatch-staging/s1fjsoup_heads.sbatch); log "two-head classifier on the jitter soup: $HJ"
st=$(wait_job $HJ); log "two-head classifier ended: $st; $(grep -o 'S1FJ_SOUP_TWOHEAD_ZIP.*' runs/sar-s1fjsoup-heads-$HJ.out | tail -1)"
log FLIPJIT_DONE
