#!/bin/bash
# Score the nearest-neighbour memorise test () on the holdout, predicting with nearest-neighbour upscaling, next
# to M stride 1 (bilinear, 337666).
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
while squeue -j  -h 2>/dev/null | grep -q .; do sleep 300; done
R=runs/ft40-s1nn-memorise-; G=results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json
[ -f $R/train/weights/best.pt ] || { echo "no weights"; exit 1; }
mkdir -p results/s1nn-memorise/one2many; cp $G results/s1nn-memorise/one2many/
python models/yolo/raw_candidates.py --weights $R/train/weights/best.pt --images $G --split-root data/course-data/train --out results/s1nn-memorise/one2many/predictions_val.json --iou 0.7 --max-det 300 --device cpu --resample nearest
python /tmp/round_hedge.py ft40-s1r-memorise-337666 s1nn-memorise
python /tmp/perclass_runs.py ft40-s1r-memorise-337666 s1nn-memorise
echo NN_EVAL_DONE
