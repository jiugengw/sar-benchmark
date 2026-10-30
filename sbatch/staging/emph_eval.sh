#!/bin/bash
# Score the emphasis tests (class weights 337406, hard targets 337459) next to the x4 memorise control, as each finishes.
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=8
for pair in "337406:cw3-memorise" "337459:hard-memorise"; do
  j=${pair%%:*}; n=${pair#*:}
  while squeue -j $j -h 2>/dev/null | grep -q .; do sleep 300; done
  r=ft40-$n-$j
  if [ -f results/$r/one2many/predictions_val.json ]; then
    python /tmp/round_hedge.py ft40-m4rep-bp4-memorise-335138 $r
    python ~/scratchDirectory/private-val/m4_curve_score.py ft40-m4rep-bp4-memorise-335138 $r
  else echo "$r: no holdout predictions ($(sacct -j $j -X -n -o State | head -1))"; fi
  echo "EMPH_DONE $n"
done
