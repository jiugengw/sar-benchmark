#!/bin/bash
# Continues flipsoup3.sh (stopped after it submitted s1adfs-memorise 338843 and s1adfs-b12 338844; its pending constant-
# speckle ingredient s1adfs-spk 338874 cancelled). Constant strong speckle lost to flips alone on the holdout (338730:
# official-mix 56.97 vs 58.32; near-copies +3.5, flipped copies -6.2, new scenes -2.2), so the three speckle ingredients
# wait for the speckle-then-clean test (338803): if it reaches flips alone (>= 58.22), they become speckle-then-clean
# flip-memorise runs (beta 8, 10, 12); otherwise plain flip-memorise runs with beta 7, 9, 11. b10 goes in at once.
# Class-selective version of flipsoup2.sh (started after its flip test won): the six all-data ingredients flip only
# images of the classes that have flipped copies in the data; aircraft, car and ship images are listed four times as
# they are. Then soup, package, round + hedge, and the stride-1 two-head classifier, as in flipsoup2.sh.
# Flip-memorise follow-up, unattended. 1) Wait for the training-split flip-memorise test (338727), score it against
# the plain memorise (337666) with flip_eval.py. 2) Only if it wins -- official-mix +0.3, or +3 mAP on the holdout's
# flipped-copy images while the other clean images lose under 0.5 (the test set has ~half copies, the holdout 16%) --
# train the six all-data soup ingredients again with --dataset-flips (same variants as the +58.00 soup; epochs over
# the 4x list from ~/sbatch-staging/flipsoup_epochs, default 20), average them into ONE model, package, round + hedge, and train the stride-1 two-head classifier on it.
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
sub() { local name=$1; shift; slot; local j=$(sbatch --parsable --time=$TL --gres=gpu:l40s:1 sbatch/finetune_detector.sbatch $name "$@"); log "$name: job $j"; }
sub s1adfs-b10 $S --tal-beta 10
EV=~/sbatch-staging/speckle_eval.out
until grep -q "^ft40-s1r-flipanneal-338803\|flipanneal-338803: no holdout" $EV 2>/dev/null; do sleep 300; done
ANN=$(awk '/^ft40-s1r-flipanneal-338803/{print $2}' $EV | tail -1)
if [ -n "$ANN" ] && [ "$(python3 -c "print(int($ANN >= 58.22))")" = 1 ]; then
  log "speckle-then-clean test: official-mix $ANN >= 58.22; three speckle-then-clean flip ingredients"
  A="--speckle-prob 0.8 --speckle-looks 2 --anneal-speckle 6"
  sub s1adfs-ann8 $S --tal-beta 8 $A; sub s1adfs-ann10 $S --tal-beta 10 $A; sub s1adfs-ann12 $S --tal-beta 12 $A
  extra="s1adfs-ann8 s1adfs-ann10 s1adfs-ann12"
else
  log "speckle-then-clean test: official-mix ${ANN:-none} < 58.22; three plain flip ingredients beta 7, 9, 11"
  sub s1adfs-b7 $S --tal-beta 7; sub s1adfs-b9 $S --tal-beta 9; sub s1adfs-b11 $S --tal-beta 11
  extra="s1adfs-b7 s1adfs-b9 s1adfs-b11"
fi
names="s1adfs-memorise s1adfs-b12 s1adfs-b10 $extra"
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
