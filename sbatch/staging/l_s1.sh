#!/bin/bash
# YOLO26l at stride 1, 100-epoch stage only (training split): compared with m at stride 1 at the same stage, to see
# whether a bigger model helps once resolution is not the limit. Starts after the all-data soup's two-head ZIP.
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
log "L_S1_STAGE1_COMPLETE $R1 (100-epoch stage only, by plan: compared with m stride 1 at the same stage)"
