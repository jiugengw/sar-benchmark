#!/bin/bash
# SPD fine-tune stages (plain -> speckle -> third (x4) -> memorise) from the finished SPD 100-epoch stage, queued
# only after the P1 test (337149) and the stride-1 + P6 test (337242) have both finished. Same recipe as spd_chain.sh.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/spd.log
log() { echo "$(date "+%m-%d %H:%M") $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d " "; }
log "SPD resume queued: waits for the SPD 100-epoch stage (337034), the P1 test (337149) and the P6 test (337242)"
for j in 337034 337149 337242; do wait_job $j > /dev/null; done
R1=runs/yolo26m-p2-dota-spd-337034
[ -f $R1/train/weights/best.pt ] || { log "no SPD best.pt, stopping"; exit 1; }
B=runs/yolo26m-p2-dota-verified-330705
run() {
  local name=$1; shift; slot
  local j=$(sbatch --parsable --time=$TIME sbatch/finetune_detector.sbatch $name "$@"); log "$name: job $j"
  local st=$(wait_job $j); log "$name ended: $st"; LAST=runs/ft40-$name-$j
  [ -f $LAST/train/weights/best.pt ] && [ -f $LAST/compute.json ] || { log "$name incomplete, stopping"; exit 1; }
}
TIME=05:00:00 run spd-plain --source $R1/train/weights/best.pt --classifier $B/image_classifier.pt --lr0 0.002 --skip-evaluation
TIME=05:00:00 run spd-speckle --source $LAST/train/weights/best.pt --classifier $B/image_classifier.pt --lr0 0.001 --speckle-prob 0.5 --skip-evaluation
TIME=06:00:00 run spd-third --source $LAST/train/weights/best.pt --classifier $B/image_classifier.pt --lr0 0.001 --repeat-classes airport,harbor,oil_tank,wind_turbine,bridge:4,playground:4 --repeat-factor 2 --repeat-width 512 --skip-evaluation
TIME=04:00:00 run spd-memorise --source $LAST/train/weights/best.pt --classifier runs/third-stage-331789-package/image_classifier.pt --no-augment --lr0 0.002 --tal-beta 8
log "SPD_COMPLETE $LAST"
