#!/bin/bash
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
W=train/weights/best.pt
R="ft40-t3-b8-336059 ft40-t3-m4spk5-336060 ft40-t3-b12-336061 ft40-t3-m4spk8-336062"
until [ $(for r in $R; do [ -f results/$r/one2many/predictions_val.json ] && echo x; done | wc -l) -eq 4 ]; do sleep 120; done
python - $(for r in $R; do echo runs/$r/$W; done) <<PY
import sys
from pathlib import Path
from swa_study import average
average([Path(p) for p in sys.argv[1:]], Path('runs/soup-bp4/t3four.pt')); print('averaged t3four', len(sys.argv) - 1)
PY
mkdir -p results/soup-bp4-t3four/one2many; cp results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json results/soup-bp4-t3four/one2many/
python models/yolo/raw_candidates.py --weights runs/soup-bp4/t3four.pt --images results/soup-bp4-t3four/one2many/instances_val_holdout.json --split-root data/course-data/train --out results/soup-bp4-t3four/one2many/predictions_val.json --iou 0.7 --max-det 300 --device cpu
python /tmp/round_hedge.py $R soup-bp4-six soup-bp4-t3four
echo T3_DONE
