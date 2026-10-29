#!/bin/bash
# Score the two speckle flip tests by holdout group as each finishes: strong constant speckle (338730), then the
# speckle-then-clean version (338803), against flips only (338727) and the plain memorise (337666).
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo OMP_NUM_THREADS=8
OUT=~/sbatch-staging/speckle_eval.out
for spec in 338730:ft40-s1r-flipspk-338730 338803:ft40-s1r-flipanneal-338803; do
  j=${spec%%:*}; run=${spec#*:}
  while squeue -j $j -h 2>/dev/null | grep -q .; do sleep 300; done
  [ -f results/$run/one2many/predictions_val.json ] || { echo "$run: no holdout predictions ($(sacct -j $j -X -n -o State | head -1))" >> $OUT; continue; }
  echo "== $(date '+%m-%d %H:%M') $run" >> $OUT
  python ~/sbatch-staging/group_runs.py ft40-s1r-memorise-337666 ft40-s1r-flipmem-338727 $run 2>&1 | grep -v -i warn >> $OUT
done
echo SPECKLE_EVAL_DONE >> $OUT
