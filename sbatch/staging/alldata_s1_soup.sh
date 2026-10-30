#!/bin/bash
# All-data STRIDE-1 soup (submission candidate), same recipe as the 54.88 all-data soup: when the all-data stride-1
# third stage (alldata_s1.sh) is done, train five memorise variants from it (L40S, batch 16, --all-data, last epoch),
# wait for the chain's own memorise run, average the six into ONE model, then package (all-data classifier), round and
# hedge the official-validation ZIP.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/alldata_s1_soup.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
until grep -q "s1ad-third ended: COMPLETED" ~/sbatch-staging/alldata_s1.log 2>/dev/null; do
  grep -q "stopping" ~/sbatch-staging/alldata_s1.log 2>/dev/null && { log "all-data chain stopped; no soup"; exit 1; }
  sleep 300
done
T=$(ls -d runs/ft40-s1ad-third-[0-9]* | head -1)/train/weights/last.pt
[ -f $T ] || { log "missing $T"; exit 1; }
CLS=runs/memorise-lr2-332326-package/image_classifier.pt
S="--source $T --classifier $CLS --all-data --skip-evaluation --no-augment --lr0 0.002 --epochs 40 --stem-stride 1 --batch 16"
M4="--speckle-classes airport,bridge,harbor,oil_tank,playground,wind_turbine --speckle-width 512 --speckle-looks 4"
sub() { local name=$1; shift; slot; local j=$(sbatch --parsable --time=05:00:00 --gres=gpu:l40s:1 sbatch/finetune_detector.sbatch $name "$@"); log "$name: job $j"; }
sub s1ad-b12 $S --tal-beta 12
sub s1ad-spk $S --tal-beta 8 --speckle-prob 0.5 --speckle-looks 4
sub s1ad-m4spk5 $S --tal-beta 8 --speckle-prob 0.5 $M4
sub s1ad-m4spk8 $S --tal-beta 8 --speckle-prob 0.8 $M4
sub s1ad-b10 $S --tal-beta 10
log "S1_SOUP_SUBMITTED"
names="s1ad-b12 s1ad-spk s1ad-m4spk5 s1ad-m4spk8 s1ad-b10 s1ad-memorise"
find_run() { ls -d runs/ft40-$1-[0-9]* 2>/dev/null | head -1; }
while :; do
  n=0; for x in $names; do r=$(find_run $x); [ -n "$r" ] && [ -f $r/compute.json ] && n=$((n+1)); done
  [ $n -eq 6 ] && break; sleep 300
done
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo
P=""; for x in $names; do P="$P $(find_run $x)/train/weights/last.pt"; done
log "averaging:$P"
mkdir -p runs/soup-s1
python - $P <<PY
import sys
from pathlib import Path
from swa_study import average
average([Path(p) for p in sys.argv[1:]], Path('runs/soup-s1/alldata-s1-six.pt')); print('averaged', len(sys.argv) - 1)
PY
cat > /tmp/s1soup_package.sbatch <<SB
#!/bin/bash
#SBATCH --job-name=sar-s1soup-pkg
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=\$PWD:\$PWD/models/yolo MPLCONFIGDIR=/tmp/s1soup-\$SLURM_JOB_ID PYTHONUNBUFFERED=1
python -u models/yolo/package_candidate.py --tag alldata-s1-soup6 --detector runs/soup-s1/alldata-s1-six.pt --curves $(find_run s1ad-memorise)/train/results.csv --box-audit results/memorise-lr2-332326/box_bias_crossfit.json --classifier-all-data
Z=\$(ls -t nonsubmissions/*/alldata-s1-soup6/alldata-s1-soup6.zip | head -1); D=\$(dirname \$Z)
python models/yolo/round_boxes.py zip \$Z \$D/alldata-s1-soup6-rounded.zip --images data/course-data/val/images.json
python models/yolo/hedge_study.py zip \$D/alldata-s1-soup6-rounded.zip \$D/alldata-s1-soup6-rounded-hedged.zip
echo "S1_SOUP_ZIP_READY \$D/alldata-s1-soup6-rounded-hedged.zip"
SB
slot; log "packaging: $(sbatch --parsable /tmp/s1soup_package.sbatch)"
