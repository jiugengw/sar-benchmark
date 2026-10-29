#!/bin/bash
# After the holdout ladder rung 341702 (~03:25 29 Oct), decide refit 4 by a rule fixed in advance (28 Oct 23:40):
# GO if official-mix (341702 vs 340514) >= +0.10 and twins >= -0.5. ~/sbatch-staging/refit4.force holding GO or NO
# (the user's Codabench result for refit 3) overrides the rule. On GO: fliprefit4.sbatch + refit_ship.sh.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/fliprefit.log; OUT=~/sbatch-staging/fliprefit_eval.out
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
log "after3.sh started: waits for the holdout rung 341702, then decides refit 4"
while ! grep -q FLIPREFIT_TEST2_DONE $OUT && squeue -j 341702 -h 2>/dev/null | grep -q .; do sleep 120; done
A=$(grep "^ft40-s1r-fliprefit-340514" $OUT | tail -1); B=$(grep "^ft40-s1r-fliprefit2-341702" $OUT | tail -1)
log "holdout rung: 340514 | $A"; log "holdout rung: 341702 | $B"
GO=NO
if [ -n "$B" ]; then
  read -r DM DT <<< $(python3 -c "a='$A'.split(); b='$B'.split(); print(f'{float(b[1])-float(a[1]):+.2f} {float(b[2])-float(a[2]):+.2f}')")
  log "rung 341702 vs 340514: official-mix $DM, twins $DT"
  python3 -c "import sys; sys.exit(0 if $DM >= 0.10 and $DT >= -0.5 else 1)" && GO=GO
fi
F=$(cat ~/sbatch-staging/refit4.force 2>/dev/null | tr -d ' \n'); [ -n "$F" ] && { log "refit4.force overrides the rule: $F"; GO=$F; }
if [ "$GO" = GO ]; then
  J=$(sbatch --parsable ~/sbatch-staging/fliprefit4.sbatch); log "refit 4 GO: job $J (L40S preferred)"
  setsid nohup bash ~/sbatch-staging/refit_ship.sh $J 4 > /dev/null 2>&1 < /dev/null &
else log "refit 4 NO (rule: official-mix >= +0.10 and twins >= -0.5)"; fi
log AFTER3_DONE
