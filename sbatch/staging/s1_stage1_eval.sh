#!/bin/bash
# When the m stride-1 100-epoch stage ends: predict the holdout with it and with the normal 100-epoch model (CPU),
# then per-class / per-size AP and official-mix for both.
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
until grep -q "s1r-plain: job" ~/sbatch-staging/reset_s1.log; do sleep 120; done
G=results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json
for pair in "s1-100:runs/yolo26m-p2-dota-s1-336150/train/weights/best.pt" "normal-100:runs/yolo26m-p2-dota-verified-330705/train/weights/best.pt"; do
  name=${pair%%:*}; w=${pair#*:}; mkdir -p results/cmp-$name/one2many; cp $G results/cmp-$name/one2many/
  python models/yolo/raw_candidates.py --weights $w --images $G --split-root data/course-data/train --out results/cmp-$name/one2many/predictions_val.json --iou 0.7 --max-det 300 --device cpu
done
python /tmp/round_hedge.py cmp-normal-100 cmp-s1-100
python ~/scratchDirectory/private-val/m4_curve_score.py cmp-normal-100 cmp-s1-100
echo STAGE1_EVAL_DONE
