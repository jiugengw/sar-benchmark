#!/bin/bash
# Within-run weight averaging of M stride 1's memorise run (337666, training split): average late checkpoints, score
# each average on the holdout (official-mix, rounded + hedged) next to the run's own best/last.
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
W=runs/ft40-s1r-memorise-337666/train/weights; G=results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json
mkdir -p runs/swa-within
for spec in "e30last:epoch30.pt last.pt" "e20e30last:epoch20.pt epoch30.pt last.pt" "last:last.pt"; do
  name=${spec%%:*}; files=""; for f in ${spec#*:}; do files="$files $W/$f"; done
  if [ "$name" = last ]; then M=$W/last.pt; else
    M=runs/swa-within/$name.pt
    python - $M $files <<PY
import sys
from pathlib import Path
from swa_study import average
average([Path(p) for p in sys.argv[2:]], Path(sys.argv[1])); print('averaged', sys.argv[2:])
PY
  fi
  mkdir -p results/swa-$name/one2many; cp $G results/swa-$name/one2many/
  python models/yolo/raw_candidates.py --weights $M --images $G --split-root data/course-data/train --out results/swa-$name/one2many/predictions_val.json --iou 0.7 --max-det 300 --device cpu
done
python /tmp/round_hedge.py ft40-s1r-memorise-337666 swa-last swa-e30last swa-e20e30last
echo SWA_WITHIN_DONE
