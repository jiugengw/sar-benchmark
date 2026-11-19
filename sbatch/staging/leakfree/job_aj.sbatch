#!/bin/bash
# Leak-free retrain, step 0: lay out the course data by results/leakfree/split.csv (tools/leakfree/make_dataset.py),
# pack it as runs/leakfree/course-data.tar (one file), put the holdout split.json where the scripts read it, then
# build a workspace (env.sh) as a smoke test. The new train split has 9,392 images like the official one, and the
# holdout is torch.randperm(9392, seed 0), so the official holdout positions (ids) apply unchanged.
#SBATCH --job-name=job-aj
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
S=~/sbatch-staging/leakfree; R=~/scratchDirectory/sar-benchmark; LF=$R/runs/leakfree; T=/tmp/lfprep-$SLURM_JOB_ID
cd $R; source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
mkdir -p $LF/runs/tm-48baseline-326894/terramind_pretrained_moelora $LF/results/yolo26m-100-p2-328339/seed_0 $T
md5sum $S/split.csv
python $S/make_dataset.py $S/split.csv data/course-data $T/course-data
python - $T/course-data data/course-data <<'PY'
import csv, hashlib, json, sys
from pathlib import Path
new, old = Path(sys.argv[1]), Path(sys.argv[2])
h = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
rows = list(csv.DictReader(open(new / 'mapping.csv')))
bad = sum(h(new / r['new_split'] / r['new_file']) != h(old / r['official_split'] / r['official_file']) for r in rows)
print('images', len(rows), 'bytes differ', bad); assert bad == 0 and len(rows) == 12883
PY
tar -C $T -cf $LF/course-data.tar course-data; rm -rf $T
A=runs/tm-48baseline-326894/terramind_pretrained_moelora/split.json; B=results/yolo26m-100-p2-328339/seed_0/split.json
python - $A $B <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:])
assert a['val_image_ids'] == b['val_image_ids'] and a['train_image_ids'] == b['train_image_ids'], 'holdouts differ'
print('holdout', len(a['val_image_ids']), 'of', len(a['val_image_ids']) + len(a['train_image_ids']))
PY
cp $A $LF/$A; cp $B $LF/$B
source $S/env.sh
du -sh $W/models $W/pretrained; tail -3 $W/prepare.log; ls data/yolo11s-sar-512
for s in train val; do echo "yolo $s: $(ls data/yolo11s-sar-512/images/$s | wc -l)"; done
python -c "import json; print({s: len(json.load(open(f'data/course-data/released-labels/{s}_instances.json'))['images']) for s in ('val','test')})"
echo LF_DATA_READY
