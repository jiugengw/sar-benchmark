#!/bin/bash
# Flip-memorise follow-up, unattended. 1) Wait for the training-split flip-memorise test (338727), score it against
# the plain memorise (337666) with flip_eval.py. 2) Only if it wins -- official-mix +0.3, or +3 mAP on the holdout's
# flipped-copy images while the other clean images lose under 0.5 (the test set has ~half copies, the holdout 16%) --
# train the six all-data soup ingredients again with --dataset-flips (same variants as the +58.00 soup, 20 epochs over
# the 4x list), average them into ONE model, package, round + hedge, and train the stride-1 two-head classifier on it.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/flipsoup.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=8
TEST=ft40-s1r-flipmem-338727
st=$(wait_job 338727); log "flip-memorise test 338727 ended: $st"
G=results/ft40-s1r-memorise-337666/one2many/instances_val_holdout.json
if [ ! -f results/$TEST/one2many/predictions_val.json ]; then
  [ -f runs/$TEST/train/weights/best.pt ] || { log "no weights for $TEST; stopping"; exit 1; }
  mkdir -p results/$TEST/one2many; cp $G results/$TEST/one2many/
  python models/yolo/raw_candidates.py --weights runs/$TEST/train/weights/best.pt --images $G --split-root data/course-data/train \
    --out results/$TEST/one2many/predictions_val.json --iou 0.7 --max-det 300 --device cpu > /dev/null 2>&1
fi
python ~/sbatch-staging/flip_eval.py ft40-s1r-memorise-337666 $TEST 2>&1 | grep -v -i warn > ~/sbatch-staging/flip_eval.out
cat ~/sbatch-staging/flip_eval.out >> $LOG
read MIX FLIP REST <<< $(grep FLIPDECISION ~/sbatch-staging/flip_eval.out | awk '{print $2, $3, $4}')
WIN=$(python3 -c "print(int($MIX >= 0.3 or ($FLIP >= 3 and $REST >= -0.5)))")
[ "$WIN" = 1 ] || { log "FLIPSOUP_NOT_STARTED: official-mix $MIX, flipped copies $FLIP, others $REST"; exit 0; }
log "FLIPSOUP_STARTING: official-mix $MIX, flipped copies $FLIP, others $REST"
T=runs/ft40-s1ad-third-338033/train/weights/last.pt
CLS=runs/memorise-lr2-332326-package/image_classifier.pt
S="--source $T --classifier $CLS --all-data --skip-evaluation --no-augment --lr0 0.002 --epochs 20 --stem-stride 1 --batch 16 --dataset-flips"
M4="--speckle-classes airport,bridge,harbor,oil_tank,playground,wind_turbine --speckle-width 512 --speckle-looks 4"
sub() { local name=$1; shift; slot; local j=$(sbatch --parsable --time=09:00:00 --gres=gpu:l40s:1 sbatch/finetune_detector.sbatch $name "$@"); log "$name: job $j"; }
sub s1adf-memorise $S --tal-beta 8
sub s1adf-b12 $S --tal-beta 12
sub s1adf-spk $S --tal-beta 8 --speckle-prob 0.5 --speckle-looks 4
sub s1adf-m4spk5 $S --tal-beta 8 --speckle-prob 0.5 $M4
sub s1adf-m4spk8 $S --tal-beta 8 --speckle-prob 0.8 $M4
sub s1adf-b10 $S --tal-beta 10
names="s1adf-memorise s1adf-b12 s1adf-spk s1adf-m4spk5 s1adf-m4spk8 s1adf-b10"
find_run() { ls -d runs/ft40-$1-[0-9]* 2>/dev/null | head -1; }
while :; do
  n=0; for x in $names; do r=$(find_run $x); [ -n "$r" ] && [ -f $r/compute.json ] && n=$((n+1)); done
  [ $n -eq 6 ] && break; sleep 300
done
P=""; for x in $names; do P="$P $(find_run $x)/train/weights/last.pt"; done
log "averaging:$P"; mkdir -p runs/soup-s1f
python - $P <<PY
import sys
from pathlib import Path
from swa_study import average
average([Path(p) for p in sys.argv[1:]], Path('runs/soup-s1f/alldata-s1f-six.pt')); print('averaged', len(sys.argv) - 1)
PY
sed -e 's/sar-s1soup-pkg/sar-s1fsoup-pkg/' -e 's#runs/soup-s1/alldata-s1-six.pt#runs/soup-s1f/alldata-s1f-six.pt#' \
    -e 's/alldata-s1-soup6/alldata-s1f-soup6/g' -e "s#runs/ft40-s1ad-memorise-[0-9]*#$(find_run s1adf-memorise)#" \
    -e 's/S1_SOUP_ZIP_READY/S1F_SOUP_ZIP_READY/' /tmp/s1soup_package.sbatch > /tmp/s1fsoup_package.sbatch
grep -q "alldata-s1f-six.pt" /tmp/s1fsoup_package.sbatch || { log "package script rewrite failed"; exit 1; }
slot; PKG=$(sbatch --parsable /tmp/s1fsoup_package.sbatch); log "packaging: $PKG"
st=$(wait_job $PKG); log "packaging ended: $st"
grep -q S1F_SOUP_ZIP_READY runs/sar-s1fsoup-pkg-$PKG.out || { log "packaging did not finish; no two-head ZIP"; exit 1; }
Z=$(grep -o "nonsubmissions/[^ ]*alldata-s1f-soup6-rounded-hedged.zip" runs/sar-s1fsoup-pkg-$PKG.out | tail -1)
sed -e 's/sar-s1soup-conv/sar-s1fsoup-heads/' -e 's#runs/soup-s1/alldata-s1-six.pt#runs/soup-s1f/alldata-s1f-six.pt#' \
    -e 's#H=runs/soup-s1/heads-338442#H=runs/soup-s1f/heads-$SLURM_JOB_ID#' \
    -e "s#nonsubmissions/251026/alldata-s1-soup6/alldata-s1-soup6-rounded-hedged.zip#$Z#g" /tmp/s1soup_conv.sbatch > /tmp/s1fsoup_heads.sbatch
sed -i "s#^python -u models/yolo/conv_head_study.py#python -u models/yolo/branch_head_study.py --detector \$DET --output \$H/branch_head.pt --epochs 20 --batch 16 --all-data\npython -u models/yolo/conv_head_study.py#" /tmp/s1fsoup_heads.sbatch
sed -i 's/S1SOUP_TWOHEAD_ZIP/S1F_SOUP_TWOHEAD_ZIP/; s/--time=06:00:00/--time=08:00:00/' /tmp/s1fsoup_heads.sbatch
grep -q "alldata-s1f-six.pt" /tmp/s1fsoup_heads.sbatch && grep -q branch_head_study /tmp/s1fsoup_heads.sbatch || { log "heads script rewrite failed"; exit 1; }
slot; log "two-head classifier on the flip soup: $(sbatch --parsable /tmp/s1fsoup_heads.sbatch)"
