#!/bin/bash
# l stride-1 100-epoch model on the holdout (CPU), scored next to the m stride-1 and normal 100-epoch models.
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
G=results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json
mkdir -p results/cmp-l-s1-100/one2many; cp $G results/cmp-l-s1-100/one2many/
python models/yolo/raw_candidates.py --weights runs/yolo26l-p2-dota-s1-336693/train/weights/best.pt --images $G --split-root data/course-data/train --out results/cmp-l-s1-100/one2many/predictions_val.json --iou 0.7 --max-det 300 --device cpu
python /tmp/round_hedge.py cmp-normal-100 cmp-s1-100 cmp-l-s1-100
python ~/scratchDirectory/private-val/m4_curve_score.py cmp-normal-100 cmp-s1-100 cmp-l-s1-100
echo L_EVAL_DONE
