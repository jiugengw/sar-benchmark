#!/bin/bash
# After the SPD 100-epoch stage: holdout per-class / official-mix next to normal, m stride 1 and l stride 1 (CPU).
cd ~/scratchDirectory/sar-benchmark
while squeue -j 337034 -h 2>/dev/null | grep -q .; do sleep 120; done
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
G=results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json
mkdir -p results/cmp-spd-100/one2many; cp $G results/cmp-spd-100/one2many/
python models/yolo/raw_candidates.py --weights runs/yolo26m-p2-dota-spd-337034/train/weights/best.pt --images $G --split-root data/course-data/train --out results/cmp-spd-100/one2many/predictions_val.json --iou 0.7 --max-det 300 --device cpu
python /tmp/round_hedge.py cmp-normal-100 cmp-s1-100 cmp-l-s1-100 cmp-spd-100
python ~/scratchDirectory/private-val/m4_curve_score.py cmp-normal-100 cmp-s1-100 cmp-l-s1-100 cmp-spd-100
echo SPD_EVAL_DONE
