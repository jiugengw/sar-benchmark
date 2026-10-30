#!/bin/bash
# Stride-1 100-epoch model with images under 512 fed at their own size (no upscaling), scored like cmp-s1-100.
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
G=results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json
mkdir -p results/cmp-s1-native/one2many; cp $G results/cmp-s1-native/one2many/
python models/yolo/raw_candidates.py --weights runs/yolo26m-p2-dota-s1-336150/train/weights/best.pt --images $G --split-root data/course-data/train --out results/cmp-s1-native/one2many/predictions_val.json --iou 0.7 --max-det 300 --device cpu --native
python /tmp/round_hedge.py cmp-normal-100 cmp-s1-100 cmp-s1-native
python ~/scratchDirectory/private-val/m4_curve_score.py cmp-normal-100 cmp-s1-100 cmp-s1-native
echo NATIVE_EVAL_DONE
