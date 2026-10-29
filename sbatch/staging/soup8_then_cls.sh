#!/bin/bash
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
W=train/weights/best.pt
until [ -f results/ft40-bp4-seed1-335844/one2many/predictions_val.json ] && [ -f results/ft40-bp4-seed2-335853/one2many/predictions_val.json ]; do sleep 60; done
SIX="runs/ft40-m4rep-bp4-memorise-335138/$W runs/ft40-bp4-beta12-335764/$W runs/ft40-bp4-memspeckle-335766/$W runs/ft40-bp4-m4speckle0.5-335815/$W runs/ft40-bp4-m4speckle0.8-335816/$W runs/bp4-polviews-memorise-335767/$W"
python - $SIX runs/ft40-bp4-seed1-335844/$W runs/ft40-bp4-seed2-335853/$W <<PY
import sys
from pathlib import Path
from swa_study import average
average([Path(p) for p in sys.argv[1:]], Path('runs/soup-bp4/eightbp4.pt')); print('averaged eightbp4', len(sys.argv) - 1)
PY
J=$(sbatch --parsable /tmp/soup_cls.sbatch runs/soup-bp4/eightbp4.pt eightbp4); echo "classifier job $J"
mkdir -p results/soup-bp4-eightbp4/one2many; cp results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json results/soup-bp4-eightbp4/one2many/
python models/yolo/raw_candidates.py --weights runs/soup-bp4/eightbp4.pt --images results/soup-bp4-eightbp4/one2many/instances_val_holdout.json --split-root data/course-data/train --out results/soup-bp4-eightbp4/one2many/predictions_val.json --iou 0.7 --max-det 300 --device cpu
python /tmp/round_hedge.py ft40-bp4-seed1-335844 ft40-bp4-seed2-335853 soup-bp4-six soup-bp4-eightbp4
echo SOUP8_DONE
