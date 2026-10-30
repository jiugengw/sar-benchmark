#!/bin/bash
# L + P1, gated on the m + P1 plain test (job 337149): if its best holdout mAP50-95 beats the normal plain stage
# (chain-dota-330938-plain, 49.0) by 0.5, train L + P1 from DOTA L weights (100 epochs, L40S, batch 16), then
# plain -> speckle -> third (x4) -> memorise on an L40S. Meanwhile score the m + P1 plain model per class on CPU.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/p1.log
log() { echo "$(date "+%m-%d %H:%M") $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d " "; }
best() { awk -F, "NR>1{if(\$9>m)m=\$9}END{print m+0}" $1/train/results.csv 2>/dev/null; }
ST=$(wait_job 337149); B1=$(best runs/ft40-p1-plain-337149); log "m + P1 plain ended: $ST, best $B1 (normal plain 0.4895, stride-1 plain 0.5055)"
( source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
  export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
  G=results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json
  for pair in "cmp-normal-plain:runs/chain-dota-330938-plain/train/weights/best.pt" "cmp-s1-plain:runs/ft40-s1r-plain-336868/train/weights/best.pt" "cmp-p1-plain:runs/ft40-p1-plain-337149/train/weights/best.pt"; do
    name=${pair%%:*}; w=${pair#*:}; mkdir -p results/$name/one2many; cp $G results/$name/one2many/
    python models/yolo/raw_candidates.py --weights $w --images $G --split-root data/course-data/train --out results/$name/one2many/predictions_val.json --iou 0.7 --max-det 300 --device cpu
  done
  python /tmp/round_hedge.py cmp-normal-plain cmp-s1-plain cmp-p1-plain
  python ~/scratchDirectory/private-val/m4_curve_score.py cmp-normal-plain cmp-s1-plain cmp-p1-plain
  echo P1_EVAL_DONE ) > ~/sbatch-staging/p1_eval.out 2>&1 &
if ! awk -v b="$B1" "BEGIN{exit !(b >= 0.495)}"; then log "m + P1 plain below 0.495: L + P1 not started"; exit 0; fi
slot; J1=$(sbatch --parsable /tmp/lp1_stage1.sbatch); R1=runs/yolo26l-p2-dota-p1-$J1; log "L + P1 100-epoch stage: job $J1 ($R1)"
STATE=$(wait_job $J1); log "L + P1 100-epoch stage ended: $STATE"
while [ "$STATE" = "TIMEOUT" ]; do
  slot; J=$(RESUME=$R1/train/weights/last.pt sbatch --parsable --export=ALL /tmp/lp1_stage1.sbatch); log "resuming: job $J"
  STATE=$(wait_job $J); log "resume ended: $STATE"
done
[ -f $R1/train/weights/best.pt ] || { log "no best.pt, stopping"; exit 1; }
B=runs/yolo26m-p2-dota-verified-330705
run() {
  local name=$1; shift; slot
  local j=$(sbatch --parsable --time=$TIME --gres=gpu:l40s:1 sbatch/finetune_detector.sbatch $name --batch 16 --skip-evaluation "$@"); log "$name: job $j (L40S)"
  local st=$(wait_job $j); log "$name ended: $st"; LAST=runs/ft40-$name-$j
  [ -f $LAST/train/weights/best.pt ] || { log "$name produced no weights, stopping"; exit 1; }
}
TIME=12:00:00 run lp1-plain --source $R1/train/weights/best.pt --classifier $B/image_classifier.pt --lr0 0.002
TIME=12:00:00 run lp1-speckle --source $LAST/train/weights/best.pt --classifier $B/image_classifier.pt --lr0 0.001 --speckle-prob 0.5
TIME=20:00:00 run lp1-third --source $LAST/train/weights/best.pt --classifier $B/image_classifier.pt --lr0 0.001 --repeat-classes airport,harbor,oil_tank,wind_turbine,bridge:4,playground:4 --repeat-factor 2 --repeat-width 512
TIME=12:00:00 run lp1-memorise --source $LAST/train/weights/best.pt --classifier $B/image_classifier.pt --no-augment --lr0 0.002 --tal-beta 8
log "LP1_COMPLETE $LAST"
