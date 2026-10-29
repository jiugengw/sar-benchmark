#!/bin/bash
# two-level soup, step 1: average the three x4 third stages, then memorise variants from the average
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo
W=train/weights/best.pt
until [ -f runs/ft40-bp4-third-seed2-335918/$W ] && ! squeue -j 335918 -h | grep -q .; do sleep 60; done
python - runs/ft40-m4rep-bp4-third-335138/$W runs/ft40-bp4-third-seed1-335917/$W runs/ft40-bp4-third-seed2-335918/$W <<PY
import sys
from pathlib import Path
from swa_study import average
average([Path(p) for p in sys.argv[1:]], Path('runs/soup-bp4/third3.pt')); print('averaged third stages', len(sys.argv) - 1)
PY
S="--source runs/soup-bp4/third3.pt --classifier runs/third-stage-331789-package/image_classifier.pt --no-augment --lr0 0.002"
M4="--speckle-classes airport,bridge,harbor,oil_tank,playground,wind_turbine --speckle-width 512 --speckle-looks 4"
sbatch --time=03:00:00 sbatch/finetune_detector.sbatch t3-b8 $S --tal-beta 8
sbatch --time=03:00:00 sbatch/finetune_detector.sbatch t3-m4spk5 $S --tal-beta 8 --speckle-prob 0.5 $M4
sbatch --time=03:00:00 sbatch/finetune_detector.sbatch t3-b12 $S --tal-beta 12
sbatch --time=03:00:00 sbatch/finetune_detector.sbatch t3-m4spk8 $S --tal-beta 8 --speckle-prob 0.8 $M4
squeue -u $USER -o "%.8i %.14j %.2t %.10M %R"
echo THIRD_SOUP_LAUNCHED
