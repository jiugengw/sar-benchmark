#!/bin/bash
# Keep both GPU slots busy for the class-selective flip soup. When one of the user's two running slots has been free
# for 15 minutes while a soup ingredient waits for an L40S, move the oldest such ingredient to a free A40 (else a free
# Blackwell slice): same run name and settings, 14 h limit (both run at about half to two-thirds L40S speed).
# A resubmission that hits the 4-submitted-jobs limit is retried until it goes through, so no ingredient is lost.
# Stops when flipsoup4.sh has submitted its last ingredient and none is waiting.
cd ~/scratchDirectory/sar-benchmark
LOG=~/sbatch-staging/flipsoup3.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
T=runs/ft40-s1ad-third-338033/train/weights/last.pt; CLS=runs/memorise-lr2-332326-package/image_classifier.pt
S="--source $T --classifier $CLS --all-data --skip-evaluation --no-augment --lr0 0.002 --epochs 20 --stem-stride 1 --batch 16 --dataset-flips --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine"
name_of() { awk -v j="$1" '$3 ~ /^s1adfs-[a-z0-9]+:$/ && $4 == "job" && $5 == j {n = $3} END {sub(":", "", n); print n}' $LOG; }
beta_of() { case $1 in s1adfs-memorise) echo 8;; s1adfs-b*) echo ${1#s1adfs-b};; *) echo "";; esac; }
gpus_free() { sinfo -n $1 -h -O "Gres:60,GresUsed:60" | awk -v t=$2 '{split($1, a, ":"); split($2, b, ":"); tot = a[3] + 0; used = b[3] + 0; print tot - used}'; }
RETRY=~/sbatch-staging/gpu_fallback.retry; touch $RETRY
free_since=""
while :; do
  while read -r name type; do  # resubmissions that failed earlier
    [ -z "$name" ] && continue
    b=$(beta_of $name); J=$(sbatch --parsable --time=14:00:00 --gres=gpu:$type:1 sbatch/finetune_detector.sbatch $name $S --tal-beta $b 2>/dev/null)
    if [ -n "$J" ]; then log "$name: job $J"; log "($name resubmitted on $type after an earlier failed attempt)"; sed -i "/^$name /d" $RETRY; fi
  done < <(cat $RETRY)
  pending=$(squeue -u $USER -h -t PD -o "%V %i %b" | grep l40s | sort | head -1 | awk '{print $2}')
  running=$(squeue -u $USER -h -t R | wc -l)
  if grep -q "s1adfs-b11: job [0-9]" $LOG && [ -z "$pending" ] && [ ! -s $RETRY ]; then log "gpu_fallback.sh: nothing left to move; stopping"; exit 0; fi
  if [ -n "$pending" ] && [ "$running" -lt 2 ]; then
    now=$(date +%s); [ -z "$free_since" ] && free_since=$now
    if [ $((now - free_since)) -ge 900 ]; then
      name=$(name_of $pending); b=$(beta_of "$name")
      if [ -z "$name" ] || [ -z "$b" ]; then log "gpu_fallback.sh: cannot tell which ingredient job $pending is; leaving it"; free_since=""; sleep 120; continue; fi
      if [ "$(gpus_free avenue a40)" -gt 0 ]; then type=a40; elif [ "$(gpus_free block x)" -gt 0 ]; then type=rtx_pro_6000_blackwell_max; else free_since=""; sleep 120; continue; fi
      scancel $pending; sleep 2
      J=$(sbatch --parsable --time=14:00:00 --gres=gpu:$type:1 sbatch/finetune_detector.sbatch $name $S --tal-beta $b 2>/dev/null)
      if [ -n "$J" ]; then log "$name: job $J"; log "($name moved from L40S job $pending to $type: a slot sat free 15 min with no L40S)"
      else echo "$name $type" >> $RETRY; log "($name: L40S job $pending cancelled, resubmission on $type refused for now; retrying)"; fi
      free_since=""
    fi
  else
    free_since=""
  fi
  sleep 120
done
