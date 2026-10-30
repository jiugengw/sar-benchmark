#!/bin/bash
# Early ZIP: the class-selective flip soup from the first FIVE ingredients (s1adfs-memorise, -b10, -b12, -b7, -b9), so it
# can be uploaded while s1adfs-b11 still trains; flipsoup5.sh builds the six-ingredient soup afterwards. Own file and
# job names (soup5, five.pt, heads5, sar-s1fs5soup-*), so the two pipelines never collide.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/flipsoup3.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=8
# log "FLIPSOUP3_STARTING: class-selective flips (aircraft, car, ship as they are), decided after the flip test (+3.59 official-mix; car -2.3, ship -3.2, aircraft -0.8)"
names="s1adfs-memorise s1adfs-b10 s1adfs-b12 s1adfs-b7 s1adfs-b9"
find_run() { ls -d runs/ft40-$1-[0-9]* 2>/dev/null | head -1; }
while :; do
  n=0; for x in $names; do r=$(find_run $x); [ -n "$r" ] && [ -f $r/compute.json ] && n=$((n+1)); done
  [ $n -eq 5 ] && break; sleep 300
done
P=""; for x in $names; do P="$P $(find_run $x)/train/weights/last.pt"; done
log "five-ingredient soup, averaging:$P"; mkdir -p runs/soup-s1fs
python - $P <<PY
import sys
from pathlib import Path
from swa_study import average
average([Path(p) for p in sys.argv[1:]], Path('runs/soup-s1fs/alldata-s1fs-five.pt')); print('averaged', len(sys.argv) - 1)
PY
sed -e 's/sar-s1soup-pkg/sar-s1fs5soup-pkg/' -e 's#runs/soup-s1/alldata-s1-six.pt#runs/soup-s1fs/alldata-s1fs-five.pt#' \
    -e 's/alldata-s1-soup6/alldata-s1fs-soup5/g' -e "s#runs/ft40-s1ad-memorise-[0-9]*#$(find_run s1adfs-memorise)#" \
    -e 's/S1_SOUP_ZIP_READY/S1FS5_SOUP_ZIP_READY/' /tmp/s1soup_package.sbatch > /tmp/s1fs5soup_package.sbatch
grep -q "alldata-s1fs-five.pt" /tmp/s1fs5soup_package.sbatch || { log "package script rewrite failed"; exit 1; }
slot; PKG=$(sbatch --parsable /tmp/s1fs5soup_package.sbatch); log "packaging: $PKG"
st=$(wait_job $PKG); log "packaging ended: $st"
grep -q S1FS5_SOUP_ZIP_READY runs/sar-s1fs5soup-pkg-$PKG.out || { log "packaging did not finish; no two-head ZIP"; exit 1; }
Z=$(grep -o "nonsubmissions/[^ ]*alldata-s1fs-soup5-rounded-hedged.zip" runs/sar-s1fs5soup-pkg-$PKG.out | tail -1)
sed -e 's/sar-s1soup-conv/sar-s1fs5soup-heads/' -e 's#runs/soup-s1/alldata-s1-six.pt#runs/soup-s1fs/alldata-s1fs-five.pt#' \
    -e 's#H=runs/soup-s1/heads-338442#H=runs/soup-s1fs/heads5-$SLURM_JOB_ID#' \
    -e "s#nonsubmissions/251026/alldata-s1-soup6/alldata-s1-soup6-rounded-hedged.zip#$Z#g" /tmp/s1soup_conv.sbatch > /tmp/s1fs5soup_heads.sbatch
sed -i "s#^python -u models/yolo/conv_head_study.py#python -u models/yolo/branch_head_study.py --detector \$DET --output \$H/branch_head.pt --epochs 20 --batch 16 --all-data\npython -u models/yolo/conv_head_study.py#" /tmp/s1fs5soup_heads.sbatch
sed -i 's/S1SOUP_TWOHEAD_ZIP/S1FS5_SOUP_TWOHEAD_ZIP/; s/--time=06:00:00/--time=08:00:00/' /tmp/s1fs5soup_heads.sbatch
grep -q "alldata-s1fs-five.pt" /tmp/s1fs5soup_heads.sbatch && grep -q branch_head_study /tmp/s1fs5soup_heads.sbatch || { log "heads script rewrite failed"; exit 1; }
slot; log "two-head classifier on the five-ingredient flip soup: $(sbatch --parsable /tmp/s1fs5soup_heads.sbatch)"
