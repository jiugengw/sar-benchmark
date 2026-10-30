#!/bin/bash
# When the five all-data soup ingredients exist: average them with the all-data beta-8 memorise run into ONE model,
# package it (all-data classifier, official-validation ZIP), then round and hedge the ZIP. Submission candidate.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/alldata_soup.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
until grep -q ALLDATA_SOUP_SUBMITTED $LOG; do sleep 300; done
names="ad-b12 ad-spk ad-m4spk5 ad-m4spk8 ad-polviews"
find_run() { ls -d runs/ft40-$1-[0-9]* 2>/dev/null | head -1; }
while :; do
  n=0; for x in $names; do r=$(find_run $x); [ -n "$r" ] && [ -f $r/compute.json ] && n=$((n+1)); done  # compute.json: written only after training ends
  [ $n -eq 5 ] && break; sleep 300
done
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo
P="runs/ft40-memorise-lr2-alldata-bp4-335303/train/weights/last.pt"; for x in $names; do P="$P $(find_run $x)/train/weights/last.pt"; done
log "averaging: $P"
python - $P <<PY
import sys
from pathlib import Path
from swa_study import average
average([Path(p) for p in sys.argv[1:]], Path('runs/soup-bp4/alldata-six.pt')); print('averaged', len(sys.argv) - 1)
PY
cat > /tmp/ad_package.sbatch <<SB
#!/bin/bash
#SBATCH --job-name=sar-ad-pkg
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=\$PWD:\$PWD/models/yolo MPLCONFIGDIR=/tmp/adpkg-\$SLURM_JOB_ID PYTHONUNBUFFERED=1
python -u models/yolo/package_candidate.py --tag alldata-soup6 --detector runs/soup-bp4/alldata-six.pt --curves runs/ft40-memorise-lr2-alldata-bp4-335303/train/results.csv --box-audit results/memorise-lr2-332326/box_bias_crossfit.json --classifier-all-data
Z=\$(ls -t nonsubmissions/*/alldata-soup6/alldata-soup6.zip | head -1); D=\$(dirname \$Z)
python models/yolo/round_boxes.py zip \$Z \$D/alldata-soup6-rounded.zip --images data/course-data/val/images.json
python models/yolo/hedge_study.py zip \$D/alldata-soup6-rounded.zip \$D/alldata-soup6-rounded-hedged.zip
echo "ALLDATA_ZIP_READY \$D/alldata-soup6-rounded-hedged.zip"
SB
while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done
log "packaging: $(sbatch --parsable /tmp/ad_package.sbatch)"
