#!/bin/bash
# average memorise runs into one model, predict the holdout on CPU, write a results folder like evaluate_export's
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
W=train/weights/best.pt
SIX="runs/ft40-m4rep-bp4-memorise-335138/$W runs/ft40-bp4-beta12-335764/$W runs/ft40-bp4-memspeckle-335766/$W runs/ft40-bp4-m4speckle0.5-335815/$W runs/ft40-bp4-m4speckle0.8-335816/$W runs/bp4-polviews-memorise-335767/$W"
TWO="runs/ft40-m4repeat-beta8-333674/$W runs/ft40-m4repeat-memorise-333334/$W"
REST="runs/ft40-bp4-mosaic-memorise-335529/$W runs/ft40-bp4-scmosaic-memorise-335707/$W runs/ft40-rot90-memorise-334368/$W runs/ft40-m4rep-noharbor-memorise-334715/$W runs/ft40-memorise-lr2-332326/$W runs/ft40-memorise-talbeta8-333238/$W runs/ft40-memorise-lr4-332325/$W runs/ft40-memorise-lr2-box15-332470/$W"
soup() {
  name=$1; shift
  python - "$name" "$@" <<PY
import sys
from pathlib import Path
from swa_study import average
average([Path(p) for p in sys.argv[2:]], Path('runs/soup-bp4') / f'{sys.argv[1]}.pt')
print('averaged', sys.argv[1], len(sys.argv) - 2)
PY
  mkdir -p results/soup-bp4-$name/one2many
  cp results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json results/soup-bp4-$name/one2many/
  python models/yolo/raw_candidates.py --weights runs/soup-bp4/$name.pt --images results/soup-bp4-$name/one2many/instances_val_holdout.json --split-root data/course-data/train --out results/soup-bp4-$name/one2many/predictions_val.json --iou 0.7 --max-det 300 --device cpu
}
soup eight $SIX $TWO
soup sixteen $SIX $TWO $REST
# sanity: the same CPU path on a model already scored on the GPU
mkdir -p results/cpucheck-six/one2many; cp results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json results/cpucheck-six/one2many/
python models/yolo/raw_candidates.py --weights runs/soup-bp4/six.pt --images results/cpucheck-six/one2many/instances_val_holdout.json --split-root data/course-data/train --out results/cpucheck-six/one2many/predictions_val.json --iou 0.7 --max-det 300 --device cpu
python /tmp/round_hedge.py soup-bp4-six cpucheck-six soup-bp4-eight soup-bp4-sixteen
echo CPU_SOUP_DONE
