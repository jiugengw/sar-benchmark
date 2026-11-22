#!/bin/bash
# Waits for the refit-2 quick-classifier ZIP (fliprefit.log line "alldata-s1fsr2: rounded + hedged ZIP: <path>") and downloads it.
H=${CLUSTER_HOST:?user@host}; K=${CLUSTER_KEY:-~/.ssh/id_ed25519}; REPO=$(cd "$(dirname "$0")/../.." && pwd)
for i in $(seq 1 60); do
  L=$(ssh -i $K -o BatchMode=yes -o ConnectTimeout=15 $H 'grep -E "alldata-s1fsr2: (rounded \+ hedged ZIP|packaging ended)|refit 2 ended" ~/sbatch-staging/fliprefit.log | tail -1' 2>/dev/null | grep -v -E "^tput:|Pseudo-terminal")
  if echo "$L" | grep -q "rounded + hedged ZIP"; then
    Z=$(echo "$L" | grep -o "nonsubmissions/.*\.zip"); mkdir -p "$REPO/$(dirname $Z)"
    scp -i $K -o BatchMode=yes $H:scratchDirectory/sar-benchmark/$Z "$REPO/$Z" && echo "R2_ZIP_FETCHED $REPO/$Z" && ls -l "$REPO/$Z"; exit 0
  elif [ -n "$L" ]; then echo "R2_FAILED $L"; exit 1; fi
  sleep 300
done
echo "R2_TIMEOUT"; exit 2
