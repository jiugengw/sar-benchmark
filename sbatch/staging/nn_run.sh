#!/bin/bash
# Nearest-neighbour upscaling test: submit when a submit slot is free (4 per user), then score it on the holdout
# (predicting with nearest-neighbour upscaling) next to M stride 1 (bilinear, 337666).
cd ~/scratchDirectory/sar-benchmark
slot() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
slot
J=$(sbatch --parsable --time=06:00:00 --gres=gpu:l40s:1 sbatch/finetune_detector.sbatch s1nn-memorise --source runs/ft40-s1r-third-337119/train/weights/best.pt --classifier runs/yolo26m-p2-dota-verified-330705/image_classifier.pt --data data/yolo-nn-512/dataset.yaml --no-augment --lr0 0.002 --tal-beta 8 --stem-stride 1 --batch 16 --skip-evaluation)
echo "$(date "+%m-%d %H:%M") nearest-neighbour test (after freeing quota): job $J" >> ~/sbatch-staging/p1.log
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
while squeue -j $J -h 2>/dev/null | grep -q .; do sleep 300; done
R=runs/ft40-s1nn-memorise-$J; G=results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json
[ -f $R/train/weights/best.pt ] || { echo "no weights ($(sacct -j $J -X -n -o State | head -1))"; exit 1; }
mkdir -p results/s1nn-memorise/one2many; cp $G results/s1nn-memorise/one2many/
python models/yolo/raw_candidates.py --weights $R/train/weights/best.pt --images $G --split-root data/course-data/train --out results/s1nn-memorise/one2many/predictions_val.json --iou 0.7 --max-det 300 --device cpu --resample nearest
python /tmp/round_hedge.py ft40-s1r-memorise-337666 s1nn-memorise
python /tmp/perclass_runs.py ft40-s1r-memorise-337666 s1nn-memorise
echo NN_EVAL_DONE
