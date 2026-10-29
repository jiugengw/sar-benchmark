#!/bin/bash
# YOLO26l at stride 1, unattended (training split, measured on the holdout): 100 epochs from DOTA l weights, then
# plain -> speckle -> third (x4) -> memorise, all at stride 1; then packaging trains an l classifier and scores the
# holdout. Starts after the all-data soup's two-head ZIP so it does not delay that.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/l_s1.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
until ls nonsubmissions/*/alldata-soup6/*-2head.zip >/dev/null 2>&1 || grep -q "Traceback" runs/sar-2head-*.out 2>/dev/null; do sleep 300; done
cat > /tmp/l_s1_stage1.sbatch <<SB
#!/bin/bash
#SBATCH --job-name=sar-l-s1-100
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=\$PWD:\$PWD/models/yolo MPLCONFIGDIR=/tmp/ls1-\$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
if [ -n "\${RESUME:-}" ]; then
  python -u -c "from ultralytics import YOLO; YOLO('\$RESUME').train(resume=True)"
else
  python -u models/yolo/train_yolo26s_p2.py --scale l --weights pretrained/yolo26l-obb.pt --train-output runs/yolo26l-p2-dota-s1-\$SLURM_JOB_ID --epochs 100 --batch 16 --workers 4 --flipud 0.0 --close-mosaic 10 --stem-stride 1
fi
echo L_S1_STAGE1_DONE
SB
slot; J1=$(sbatch --parsable /tmp/l_s1_stage1.sbatch); R1=runs/yolo26l-p2-dota-s1-$J1; log "l 100-epoch stage at stride 1: job $J1 ($R1)"
STATE=$(wait_job $J1); log "l 100-epoch stage ended: $STATE"
while [ "$STATE" = "TIMEOUT" ]; do
  slot; J=$(RESUME=$R1/train/weights/last.pt sbatch --parsable --export=ALL /tmp/l_s1_stage1.sbatch); log "resuming: job $J"
  STATE=$(wait_job $J); log "resume ended: $STATE"
done
[ -f $R1/train/weights/best.pt ] || { log "no best.pt, stopping"; exit 1; }
CLS=runs/yolo26m-p2-dota-verified-330705/image_classifier.pt  # required argument only; every stage skips evaluation
run() {
  local name=$1; shift; slot
  local j=$(sbatch --parsable --time=$TIME sbatch/finetune_detector.sbatch $name "$@"); log "$name: job $j"
  local st=$(wait_job $j); log "$name ended: $st"; LAST=runs/ft40-$name-$j
  [ -f $LAST/train/weights/best.pt ] && [ -f $LAST/compute.json ] || { log "$name incomplete, stopping"; exit 1; }
}
TIME=16:00:00 run ls1-plain --source $R1/train/weights/best.pt --classifier $CLS --lr0 0.002 --stem-stride 1 --skip-evaluation
TIME=16:00:00 run ls1-speckle --source $LAST/train/weights/best.pt --classifier $CLS --lr0 0.001 --speckle-prob 0.5 --stem-stride 1 --skip-evaluation
TIME=24:00:00 run ls1-third --source $LAST/train/weights/best.pt --classifier $CLS --lr0 0.001 --stem-stride 1 --repeat-classes airport,harbor,oil_tank,wind_turbine,bridge:4,playground:4 --repeat-factor 2 --repeat-width 512 --skip-evaluation
TIME=16:00:00 run ls1-memorise --source $LAST/train/weights/best.pt --classifier $CLS --no-augment --lr0 0.002 --tal-beta 8 --stem-stride 1 --skip-evaluation
slot; J=$(sbatch --parsable --time=04:00:00 sbatch/package_candidate.sbatch ls1-memorise $LAST/train/weights/best.pt $LAST/train/results.csv); log "packaging (l classifier + holdout scores): job $J -> results/ls1-memorise"
wait_job $J >/dev/null; log "L_S1_COMPLETE"
