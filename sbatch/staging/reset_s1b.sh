#!/bin/bash
# Takes over from reset_s1.sh after its 100-epoch stage (job 336150, Blackwell slice): the four fine-tune stages run on
# a full L40S (48 GB) at batch 16, the same batch as the normal-resolution chain, ~2.5x faster than a Blackwell slice.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/reset_s1.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
B=runs/yolo26m-p2-dota-verified-330705
R1=runs/yolo26m-p2-dota-s1-336150
log "driver moved to reset_s1b.sh: fine-tune stages will run on an L40S at batch 16"
STATE=$(wait_job 336150); log "100-epoch stage ended: $STATE"
while [ "$STATE" = "TIMEOUT" ]; do
  slot; J=$(RESUME=$R1/train/weights/last.pt sbatch --parsable --export=ALL /tmp/s1_stage1.sbatch); log "resuming 100-epoch stage: job $J"
  STATE=$(wait_job $J); log "resume ended: $STATE"
done
[ -f $R1/train/weights/best.pt ] || { log "no best.pt, stopping"; exit 1; }
run() {  # name, then finetune_detector options
  local name=$1; shift; slot
  local j=$(sbatch --parsable --time=$TIME --gres=gpu:l40s:1 sbatch/finetune_detector.sbatch $name --batch 16 "$@"); log "$name: job $j (L40S)"
  local st=$(wait_job $j); log "$name ended: $st"; LAST=runs/ft40-$name-$j
  [ -f $LAST/train/weights/best.pt ] || { log "$name produced no weights, stopping"; exit 1; }
}
TIME=10:00:00 run s1r-plain --source $R1/train/weights/best.pt --classifier $B/image_classifier.pt --lr0 0.002 --stem-stride 1 --skip-evaluation
TIME=10:00:00 run s1r-speckle --source $LAST/train/weights/best.pt --classifier $B/image_classifier.pt --lr0 0.001 --speckle-prob 0.5 --stem-stride 1 --skip-evaluation
TIME=16:00:00 run s1r-third --source $LAST/train/weights/best.pt --classifier $B/image_classifier.pt --lr0 0.001 --stem-stride 1 --repeat-classes airport,harbor,oil_tank,wind_turbine,bridge:4,playground:4 --repeat-factor 2 --repeat-width 512 --skip-evaluation
TIME=10:00:00 run s1r-memorise --source $LAST/train/weights/best.pt --classifier runs/third-stage-331789-package/image_classifier.pt --no-augment --lr0 0.002 --tal-beta 8 --stem-stride 1
log "RESET_S1_COMPLETE $LAST"
