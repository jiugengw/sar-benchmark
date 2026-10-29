#!/bin/bash
# Score the two 140-epoch straight runs on the holdout (official-mix, rounded + hedged, per class) next to the normal
# 100-epoch + plain-40 chain (cmp-normal-plain).
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=16
G=results/ft40-m4rep-bp4-memorise-335138/one2many/instances_val_holdout.json
names=""
for spec in "338487:auto" "338488:0.02"; do
  j=${spec%%:*}; lr=${spec#*:}
  while squeue -j $j -h 2>/dev/null | grep -q .; do sleep 300; done
  R=runs/yolo26m-p2-dota-140-$lr-$j
  [ -f $R/train/weights/best.pt ] || { echo "$R: no weights ($(sacct -j $j -X -n -o State | head -1))"; continue; }
  n=cmp-m140-$lr; mkdir -p results/$n/one2many; cp $G results/$n/one2many/
  python models/yolo/raw_candidates.py --weights $R/train/weights/best.pt --images $G --split-root data/course-data/train --out results/$n/one2many/predictions_val.json --iou 0.7 --max-det 300 --device cpu
  echo "$R best raw: $(awk -F, 'NR>1{if($9>m){m=$9;e=$1}}END{print e, m}' $R/train/results.csv)"
  names="$names $n"
done
python /tmp/round_hedge.py cmp-normal-plain $names
python /tmp/perclass_runs.py cmp-normal-plain $names
echo M140_EVAL_DONE
