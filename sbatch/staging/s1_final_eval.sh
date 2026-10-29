#!/bin/bash
# Score M stride 1's memorise stage (337666) against the normal x4 memorise control, then the P6 third-stage curve.
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=8
best() { awk -F, 'NR>1{if($9>m){m=$9;e=$1}}END{print e, m}' "$1"; }
while squeue -j 337666 -h 2>/dev/null | grep -q .; do sleep 300; done
r=ft40-s1r-memorise-337666
if [ -f results/$r/one2many/predictions_val.json ]; then
  python /tmp/round_hedge.py ft40-m4rep-bp4-memorise-335138 $r
  python ~/scratchDirectory/private-val/m4_curve_score.py ft40-m4rep-bp4-memorise-335138 $r
else echo "$r: no holdout predictions ($(sacct -j 337666 -X -n -o State | head -1))"; fi
echo "P6 third best: $(best runs/ft40-s1p6-third-337242/train/results.csv) vs stride-1 third best: $(best runs/ft40-s1r-third-337119/train/results.csv)"
echo S1_FINAL_DONE
