#!/bin/bash
# Stride-1 reset, unattended: 100 epochs from DOTA weights at stride 1, then plain -> speckle -> third (x4) -> memorise,
# all at stride 1. Each stage is its own job (24 h limit); a stage is submitted when the previous one has finished and
# a submit slot is free (4 per user). The 100-epoch stage resumes from last.pt if it ever hits its time limit.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/reset_s1.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
B=runs/yolo26m-p2-dota-verified-330705
cat > /tmp/s1_stage1.sbatch <<SB
#!/bin/bash
#SBATCH --job-name=sar-s1-100
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=\$PWD:\$PWD/models/yolo MPLCONFIGDIR=/tmp/s1-\$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
if [ -n "\${RESUME:-}" ]; then
  python -u -c "from ultralytics import YOLO; YOLO('\$RESUME').train(resume=True)"
else
  python -u models/yolo/train_yolo26s_p2.py --scale m --weights pretrained/yolo26m-obb.pt --train-output runs/yolo26m-p2-dota-s1-\$SLURM_JOB_ID --epochs 100 --batch 16 --workers 4 --flipud 0.0 --close-mosaic 10 --stem-stride 1
fi
echo S1_STAGE1_DONE
SB
slot; J1=$(sbatch --parsable /tmp/s1_stage1.sbatch); R1=runs/yolo26m-p2-dota-s1-$J1; log "100-epoch stage at stride 1: job $J1 ($R1)"
STATE=$(wait_job $J1); log "100-epoch stage ended: $STATE"
while [ "$STATE" = "TIMEOUT" ]; do
  slot; J=$(RESUME=$R1/train/weights/last.pt sbatch --parsable --export=ALL /tmp/s1_stage1.sbatch); log "resuming 100-epoch stage: job $J"
  STATE=$(wait_job $J); log "resume ended: $STATE"
done
[ -f $R1/train/weights/best.pt ] || { log "no best.pt, stopping"; exit 1; }
run() {  # name, then finetune_detector options
  local name=$1; shift; slot
  local j=$(sbatch --parsable --time=$TIME sbatch/finetune_detector.sbatch $name "$@"); log "$name: job $j"
  local st=$(wait_job $j); log "$name ended: $st"; LAST=runs/ft40-$name-$j
  [ -f $LAST/train/weights/best.pt ] || { log "$name produced no weights, stopping"; exit 1; }
}
TIME=10:00:00 run s1r-plain --source $R1/train/weights/best.pt --classifier $B/image_classifier.pt --lr0 0.002 --stem-stride 1 --skip-evaluation
TIME=10:00:00 run s1r-speckle --source $LAST/train/weights/best.pt --classifier $B/image_classifier.pt --lr0 0.001 --speckle-prob 0.5 --stem-stride 1 --skip-evaluation
TIME=16:00:00 run s1r-third --source $LAST/train/weights/best.pt --classifier $B/image_classifier.pt --lr0 0.001 --stem-stride 1 --repeat-classes airport,harbor,oil_tank,wind_turbine,bridge:4,playground:4 --repeat-factor 2 --repeat-width 512 --skip-evaluation
TIME=10:00:00 run s1r-memorise --source $LAST/train/weights/best.pt --classifier runs/third-stage-331789-package/image_classifier.pt --no-augment --lr0 0.002 --tal-beta 8 --stem-stride 1
log "RESET_S1_COMPLETE $LAST"
