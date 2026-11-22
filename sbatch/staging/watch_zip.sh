#!/bin/bash
# watch_zip.sh TAG quick|twohead TRIES: waits for TAG's ZIP line in fliprefit.log and downloads the ZIP.
TAG=$1; KIND=$2; N=${3:-80}
H=${CLUSTER_HOST:?user@host}; K=${CLUSTER_KEY:-~/.ssh/id_ed25519}; REPO=$(cd "$(dirname "$0")/../.." && pwd)
if [ $KIND = quick ]; then PAT="$TAG: (rounded \+ hedged ZIP|packaging ended)|refit 3 ended"; OK="rounded + hedged ZIP"
else PAT="$TAG: (two-head classifier ended|packaging ended)|refit 3 ended"; OK="S1FSR_TWOHEAD_ZIP"; fi
for i in $(seq 1 $N); do
  L=$(ssh -i $K -o BatchMode=yes -o ConnectTimeout=15 $H "grep -E '$PAT' ~/sbatch-staging/fliprefit.log | tail -1" 2>/dev/null | grep -v -E "^tput:|Pseudo-terminal")
  if echo "$L" | grep -q "$OK"; then
    Z=$(echo "$L" | grep -o "nonsubmissions/[^ ]*\.zip" | tail -1); mkdir -p "$REPO/$(dirname $Z)"
    scp -i $K -o BatchMode=yes $H:scratchDirectory/sar-benchmark/$Z "$REPO/$Z" && echo "ZIP_FETCHED $REPO/$Z" && echo "$L" && ls -l "$REPO/$Z"; exit 0
  elif [ -n "$L" ]; then echo "FAILED $L"; exit 1; fi
  sleep 300
done
echo TIMEOUT; exit 2
