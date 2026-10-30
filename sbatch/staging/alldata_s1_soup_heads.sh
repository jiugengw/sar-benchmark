#!/bin/bash
# After the all-data stride-1 soup ZIP is ready: train the stride-1 two-head classifier (branch copy of backbone layers
# 5-10, 20 epochs; conv head, 30 epochs) on the SOUP model's features with all data (last epoch kept), predict the
# official validation images, and write a second ZIP whose image labels are 0.5 branch + 0.5 conv (size rule); boxes
# unchanged. One model: the soup detector plus its own heads.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/alldata_s1_soup.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
until grep -q "packaging:" $LOG 2>/dev/null; do sleep 300; done
PKG=$(grep -o "packaging: [0-9]*" $LOG | tail -1 | awk '{print $2}')
while squeue -j $PKG -h 2>/dev/null | grep -q .; do sleep 300; done
grep -q S1_SOUP_ZIP_READY runs/sar-s1soup-pkg-$PKG.out || { log "soup packaging $PKG did not finish; no two-head ZIP"; exit 1; }
Z=$(grep -o "nonsubmissions/[^ ]*alldata-s1-soup6-rounded-hedged.zip" runs/sar-s1soup-pkg-$PKG.out | tail -1)
cat > /tmp/s1soup_heads.sbatch <<SB
#!/bin/bash
#SBATCH --job-name=sar-s1soup-heads
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=\$PWD:\$PWD/models/yolo MPLCONFIGDIR=/tmp/s1sh-\$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
DET=runs/soup-s1/alldata-s1-six.pt; H=runs/soup-s1/heads-\$SLURM_JOB_ID
python -u models/yolo/branch_head_study.py --detector \$DET --output \$H/branch_head.pt --epochs 20 --batch 16 --all-data
python -u models/yolo/conv_head_study.py --detector \$DET --output \$H/conv_head.pt --batch 16 --epochs 50 --all-data
python -u models/yolo/branch_head_predict.py --detector \$DET --branch \$H/branch_head.pt --images data/course-data/val/images.json --split-root data/course-data/val --out \$H/val_probs_branch.json
python -u models/yolo/conv_head_predict.py --detector \$DET --branch \$H/conv_head.pt --images data/course-data/val/images.json --split-root data/course-data/val --out \$H/val_probs_conv.json
python - <<PY
import json, zipfile, sys
import numpy as np
sys.path.insert(0, 'models/yolo')
from classifier_combo_study import apply_size_rule, size_allowed
images = json.load(open('data/course-data/val/images.json'))['images']; ids = [im['id'] for im in images]
b = json.load(open('\$H/val_probs_branch.json')); c = json.load(open('\$H/val_probs_conv.json'))
P = apply_size_rule(0.5 * np.array([b[str(i)] for i in ids]) + 0.5 * np.array([c[str(i)] for i in ids]), [(im['width'], im['height']) for im in images], size_allowed())
labels = P.argmax(1)
src = zipfile.ZipFile('$Z')
old = dict(l.split(',') for l in src.read('classification.csv').decode().strip().splitlines()[1:])
target = '$Z'.replace('-rounded-hedged.zip', '-rounded-hedged-s1two.zip')
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as out:
    out.writestr('detection.json', src.read('detection.json'))
    out.writestr('classification.csv', 'image_id,label\n' + ''.join(f'{i},{int(k)}\n' for i, k in zip(ids, labels)))
print('S1SOUP_TWOHEAD_ZIP', target, 'labels changed vs the mean-head ZIP:', sum(old[str(i)] != str(int(k)) for i, k in zip(ids, labels)))
PY
SB
while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done
log "two-head classifier on the soup: $(sbatch --parsable /tmp/s1soup_heads.sbatch)"
