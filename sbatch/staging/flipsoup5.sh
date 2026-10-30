#!/bin/bash
# Replaces flipsoup4.sh + gpu_fallback.sh for the class-selective flip soup. The other L40S cards are held by other
# users for 16-23 h, so the remaining ingredients run as (1) ONE chain job on an L40S, s1adfs-b12 -> b7 -> b9 back to
# back (24 h limit, the GPU is never released between them), and (2) s1adfs-b11 alone on a Blackwell slice (14 h).
# Fallback: if the chain has not started 30 min after s1adfs-memorise (338843) ends, it moves to a Blackwell slice as
# b12 -> b7 (24 h) plus b9 alone (14 h). Then, as before: average the six into ONE model, package, round + hedge, and
# train the stride-1 two-head classifier on it (L40S).
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/flipsoup3.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=8
# log "FLIPSOUP3_STARTING: class-selective flips (aircraft, car, ship as they are), decided after the flip test (+3.59 official-mix; car -2.3, ship -3.2, aircraft -0.8)"
EP=$(cat ~/sbatch-staging/flipsoup_epochs 2>/dev/null || echo 20)  # read at launch, so it can be changed while waiting
TL=$(printf "%02d:00:00" $((EP * 3 / 10 + 3)))
log "soup ingredients: $EP epochs over the 4x list, time limit $TL"
T=runs/ft40-s1ad-third-338033/train/weights/last.pt
CLS=runs/memorise-lr2-332326-package/image_classifier.pt
S="--source $T --classifier $CLS --all-data --skip-evaluation --no-augment --lr0 0.002 --epochs $EP --stem-stride 1 --batch 16 --dataset-flips --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine"
CHAIN=~/sbatch-staging/flip_chain.sbatch
BW=rtx_pro_6000_blackwell_max
chain() { local gpu=$1 tl=$2; shift 2; LIST="$*" sbatch --parsable --export=ALL --gres=gpu:$gpu:1 --time=$tl $CHAIN; }
single() { local name=$1 b=$2; sbatch --parsable --time=14:00:00 --gres=gpu:$BW:1 sbatch/finetune_detector.sbatch $name $S --tal-beta $b; }
L=""; B11=""; LB=""; B9=""
until [ -n "$L" ]; do L=$(chain l40s 24:00:00 s1adfs-b12:12 s1adfs-b7:7 s1adfs-b9:9); [ -z "$L" ] && sleep 60; done
log "chain job on L40S: s1adfs-b12 -> b7 -> b9: job $L"
until [ -n "$B11" ]; do B11=$(single s1adfs-b11 11); [ -z "$B11" ] && sleep 60; done
log "s1adfs-b11: job $B11 (Blackwell slice, 14 h)"
wait_job 338843 > /dev/null
sleep 1800
if squeue -j $L -h -t PD 2>/dev/null | grep -q .; then
  scancel $L; sleep 5
  until [ -n "$LB" ]; do LB=$(chain $BW 24:00:00 s1adfs-b12:12 s1adfs-b7:7); [ -z "$LB" ] && sleep 60; done
  log "chain $L still waiting for an L40S 30 min after 338843 ended: cancelled; s1adfs-b12 -> b7 on a Blackwell slice: job $LB"
  until [ -n "$B9" ]; do B9=$(single s1adfs-b9 9); [ -z "$B9" ] && sleep 120; done
  log "s1adfs-b9: job $B9 (Blackwell slice, 14 h)"
else
  log "chain job $L started on an L40S"
fi
names="s1adfs-memorise s1adfs-b12 s1adfs-b10 s1adfs-b7 s1adfs-b9 s1adfs-b11"
find_run() { ls -d runs/ft40-$1-[0-9]* 2>/dev/null | head -1; }
while :; do
  n=0; for x in $names; do r=$(find_run $x); [ -n "$r" ] && [ -f $r/compute.json ] && n=$((n+1)); done
  [ $n -eq 6 ] && break; sleep 300
done
P=""; for x in $names; do P="$P $(find_run $x)/train/weights/last.pt"; done
log "averaging:$P"; mkdir -p runs/soup-s1fs
python - $P <<PY
import sys
from pathlib import Path
from swa_study import average
average([Path(p) for p in sys.argv[1:]], Path('runs/soup-s1fs/alldata-s1fs-six.pt')); print('averaged', len(sys.argv) - 1)
PY
sed -e 's/sar-s1soup-pkg/sar-s1fssoup-pkg/' -e 's#runs/soup-s1/alldata-s1-six.pt#runs/soup-s1fs/alldata-s1fs-six.pt#' \
    -e 's/alldata-s1-soup6/alldata-s1fs-soup6/g' -e "s#runs/ft40-s1ad-memorise-[0-9]*#$(find_run s1adfs-memorise)#" \
    -e 's/S1_SOUP_ZIP_READY/S1FS_SOUP_ZIP_READY/' /tmp/s1soup_package.sbatch > /tmp/s1fssoup_package.sbatch
grep -q "alldata-s1fs-six.pt" /tmp/s1fssoup_package.sbatch || { log "package script rewrite failed"; exit 1; }
slot; PKG=$(sbatch --parsable /tmp/s1fssoup_package.sbatch); log "packaging: $PKG"
st=$(wait_job $PKG); log "packaging ended: $st"
grep -q S1FS_SOUP_ZIP_READY runs/sar-s1fssoup-pkg-$PKG.out || { log "packaging did not finish; no two-head ZIP"; exit 1; }
Z=$(grep -o "nonsubmissions/[^ ]*alldata-s1fs-soup6-rounded-hedged.zip" runs/sar-s1fssoup-pkg-$PKG.out | tail -1)
sed -e 's/sar-s1soup-conv/sar-s1fssoup-heads/' -e 's#runs/soup-s1/alldata-s1-six.pt#runs/soup-s1fs/alldata-s1fs-six.pt#' \
    -e 's#H=runs/soup-s1/heads-338442#H=runs/soup-s1fs/heads-$SLURM_JOB_ID#' \
    -e "s#nonsubmissions/251026/alldata-s1-soup6/alldata-s1-soup6-rounded-hedged.zip#$Z#g" /tmp/s1soup_conv.sbatch > /tmp/s1fssoup_heads.sbatch
sed -i "s#^python -u models/yolo/conv_head_study.py#python -u models/yolo/branch_head_study.py --detector \$DET --output \$H/branch_head.pt --epochs 20 --batch 16 --all-data\npython -u models/yolo/conv_head_study.py#" /tmp/s1fssoup_heads.sbatch
sed -i 's/S1SOUP_TWOHEAD_ZIP/S1FS_SOUP_TWOHEAD_ZIP/; s/--time=06:00:00/--time=08:00:00/' /tmp/s1fssoup_heads.sbatch
grep -q "alldata-s1fs-six.pt" /tmp/s1fssoup_heads.sbatch && grep -q branch_head_study /tmp/s1fssoup_heads.sbatch || { log "heads script rewrite failed"; exit 1; }
slot; log "two-head classifier on the flip soup: $(sbatch --parsable /tmp/s1fssoup_heads.sbatch)"
