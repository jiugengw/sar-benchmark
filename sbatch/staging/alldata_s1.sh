#!/bin/bash
# All-data stride-1 chain (submission candidate, run alongside the train-split chain that measures it):
# from the stride-1 100-epoch model, plain -> speckle -> third (x4) -> memorise, each --all-data (last epoch kept),
# then package (all-data classifier), round and hedge the ZIP.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/alldata_s1.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 300; done; sacct -j $1 -X -n -o State | head -1 | tr -d ' '; }
until grep -q "s1r-plain: job" ~/sbatch-staging/reset_s1.log 2>/dev/null; do sleep 300; done  # 100-epoch stage finished
R1=$(grep -o "runs/yolo26m-p2-dota-s1-[0-9]*" ~/sbatch-staging/reset_s1.log | head -1)
[ -f $R1/train/weights/best.pt ] || { log "no stride-1 100-epoch weights"; exit 1; }
CLS=runs/memorise-lr2-332326-package/image_classifier.pt
run() {
  local name=$1; shift; slot
  local j=$(sbatch --parsable --time=$TIME --gres=gpu:l40s:1 sbatch/finetune_detector.sbatch $name --batch 16 "$@"); log "$name: job $j"
  local st=$(wait_job $j); log "$name ended: $st"; LAST=runs/ft40-$name-$j
  [ -f $LAST/train/weights/last.pt ] && [ -f $LAST/compute.json ] || { log "$name incomplete, stopping"; exit 1; }
}
TIME=12:00:00 run s1ad-plain --source $R1/train/weights/best.pt --classifier $CLS --all-data --skip-evaluation --lr0 0.002 --stem-stride 1
TIME=12:00:00 run s1ad-speckle --source $LAST/train/weights/last.pt --classifier $CLS --all-data --skip-evaluation --lr0 0.001 --speckle-prob 0.5 --stem-stride 1
TIME=18:00:00 run s1ad-third --source $LAST/train/weights/last.pt --classifier $CLS --all-data --skip-evaluation --lr0 0.001 --stem-stride 1 --repeat-classes airport,harbor,oil_tank,wind_turbine,bridge:4,playground:4 --repeat-factor 2 --repeat-width 512
TIME=12:00:00 run s1ad-memorise --source $LAST/train/weights/last.pt --classifier $CLS --all-data --skip-evaluation --no-augment --lr0 0.002 --tal-beta 8 --stem-stride 1
cat > /tmp/s1ad_package.sbatch <<SB
#!/bin/bash
#SBATCH --job-name=sar-s1ad-pkg
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=\$PWD:\$PWD/models/yolo MPLCONFIGDIR=/tmp/s1adpkg-\$SLURM_JOB_ID PYTHONUNBUFFERED=1
python -u models/yolo/package_candidate.py --tag alldata-s1 --detector $LAST/train/weights/last.pt --curves $LAST/train/results.csv --box-audit results/memorise-lr2-332326/box_bias_crossfit.json --classifier-all-data
Z=\$(ls -t nonsubmissions/*/alldata-s1/alldata-s1.zip | head -1); D=\$(dirname \$Z)
python models/yolo/round_boxes.py zip \$Z \$D/alldata-s1-rounded.zip --images data/course-data/val/images.json
python models/yolo/hedge_study.py zip \$D/alldata-s1-rounded.zip \$D/alldata-s1-rounded-hedged.zip
echo "ALLDATA_S1_ZIP_READY \$D/alldata-s1-rounded-hedged.zip"
SB
slot; log "packaging: $(sbatch --parsable /tmp/s1ad_package.sbatch)"
