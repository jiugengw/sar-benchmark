#!/bin/bash
# SPD first layer (spd.py), full chain at normal grid sizes, training split, measured on the holdout:
# 100 epochs from DOTA m weights -> plain -> speckle -> third (x4) -> memorise. Starts after l's 100-epoch stage.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/spd.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
until grep -q "L_S1_STAGE1_COMPLETE\|stopping" ~/sbatch-staging/l_s1.log 2>/dev/null; do sleep 300; done
cat > /tmp/spd_stage1.sbatch <<SB
#!/bin/bash
#SBATCH --job-name=sar-spd-100
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=\$PWD:\$PWD/models/yolo MPLCONFIGDIR=/tmp/spd-\$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python -u models/yolo/train_yolo26s_p2.py --scale m --weights pretrained/yolo26m-obb.pt --train-output runs/yolo26m-p2-dota-spd-\$SLURM_JOB_ID --epochs 100 --batch 16 --workers 4 --flipud 0.0 --close-mosaic 10 --spd-stem
SB
slot; J1=$(sbatch --parsable /tmp/spd_stage1.sbatch); R1=runs/yolo26m-p2-dota-spd-$J1; log "SPD 100-epoch stage: job $J1 ($R1)"
STATE=$(wait_job $J1); log "SPD 100-epoch stage ended: $STATE"
[ -f $R1/train/weights/best.pt ] || { log "no best.pt, stopping"; exit 1; }
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
