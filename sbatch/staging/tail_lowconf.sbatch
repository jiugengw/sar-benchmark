#!/bin/bash
# Low-score tail test (28 Oct): the holdout refit
# 340514 (training-split twin of refit 2) predicted again with conf 1e-5 and up to 1000 boxes per image, then
# tools/analysis/lowconf_tail_study.py scores the sub-0.001 boxes appended at three places in the ranking.
#SBATCH --job-name=sar-tail-lowconf
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --time=01:30:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-tail-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
RUN=ft40-s1r-fliprefit-340514; R=results/tail-lowconf-340514; OUT=~/sbatch-staging/tail_eval.out
mkdir -p $R; cp results/$RUN/one2many/instances_val_holdout.json $R/
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python - <<'PY'
import json, subprocess, sys, tempfile, zipfile
from pathlib import Path
R = Path('results/tail-lowconf-340514'); gt = json.load(open(R / 'instances_val_holdout.json'))
with tempfile.TemporaryDirectory(prefix='sar-tail-') as t:
    view = Path(t) / 'data'; (view / 'train').mkdir(parents=True); (view / 'val').mkdir()
    (view / 'train/labels.csv').symlink_to(Path('data/course-data/train/labels.csv').resolve())
    for im in gt['images']:
        target = view / 'val' / im['file_name']; target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(Path('data/course-data/train', im['file_name']).resolve())
    (view / 'val/images.json').write_text(json.dumps({'images': gt['images']}))
    z = Path(t) / 'lowconf.zip'
    subprocess.run([sys.executable, '-u', 'models/yolo/make_submission.py', '--weights',
                    'runs/ft40-s1r-fliprefit-340514/train/weights/best.pt', '--detection-branch', 'one2many',
                    '--data', str(view), '--out', str(z), '--conf', '1e-5', '--max-det', '1000'], check=True)
    p = json.loads(zipfile.ZipFile(z).read('detection.json'))
(R / 'predictions_lowconf.json').write_text(json.dumps(p)); print('LOWCONF_BOXES', len(p))
PY
python ~/sbatch-staging/newids.py $RUN $R/new_scene_ids.json 2>&1 | grep -v -i warn
{ echo "== $(date '+%m-%d %H:%M') low-score tail test on $RUN (job $SLURM_JOB_ID)"
  python tools/analysis/lowconf_tail_study.py $R 2>&1 | grep -v -i -E "warn|_methods|rcount"; } >> $OUT
echo TAIL_DONE >> $OUT
