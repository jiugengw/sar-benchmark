#!/bin/bash
# All-data version of the 6-run soup (submission candidate): memorise variants from the all-data x4 third stage
# (job 335303, trained on training split + holdout), each --all-data, keeping the last epoch.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/alldata_soup.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
T=runs/ft40-third-alldata-bp4-335303/train/weights/last.pt
CLS=runs/memorise-lr2-332326-package/image_classifier.pt
S="--source $T --classifier $CLS --all-data --skip-evaluation --no-augment --lr0 0.002 --epochs 40"
M4="--speckle-classes airport,bridge,harbor,oil_tank,playground,wind_turbine --speckle-width 512 --speckle-looks 4"
[ -f $T ] || { log "missing $T"; exit 1; }
slot; log "beta12: $(sbatch --parsable --time=03:00:00 sbatch/finetune_detector.sbatch ad-b12 $S --tal-beta 12)"
slot; log "speckle all: $(sbatch --parsable --time=03:00:00 sbatch/finetune_detector.sbatch ad-spk $S --tal-beta 8 --speckle-prob 0.5 --speckle-looks 4)"
slot; log "m4 speckle 0.5: $(sbatch --parsable --time=03:00:00 sbatch/finetune_detector.sbatch ad-m4spk5 $S --tal-beta 8 --speckle-prob 0.5 $M4)"
slot; log "m4 speckle 0.8: $(sbatch --parsable --time=03:00:00 sbatch/finetune_detector.sbatch ad-m4spk8 $S --tal-beta 8 --speckle-prob 0.8 $M4)"
V=runs/pol-views-335767
cat > /tmp/ad_polviews.sbatch <<SB
#!/bin/bash
#SBATCH --job-name=sar-ad-pv
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=\$PWD:\$PWD/models/yolo MPLCONFIGDIR=/tmp/adpv-\$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python -u models/yolo/train_with_extra.py --extra-zip $V/views.zip --extra-labels $V/views_labels.pkl --tag ft40-ad-polviews-\$SLURM_JOB_ID $S --tal-beta 8
SB
slot; log "polviews: $(sbatch --parsable /tmp/ad_polviews.sbatch)"
log "ALLDATA_SOUP_SUBMITTED"
