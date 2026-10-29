#!/bin/bash
# watch_tail3.sh N TRIES: waits for the N-th "alldata-s1fsr3: tail ZIP:" line in fliprefit.log and downloads that ZIP.
N=$1; T=${2:-90}
H=jgwong.2023@10.193.104.102; K=~/.ssh/id_ed25519_smu; REPO=/Users/jiugeng/Projects/sar-benchmark
for i in $(seq 1 $T); do
  L=$(ssh -i $K -o BatchMode=yes -o ConnectTimeout=15 $H "grep 'alldata-s1fsr3: tail ZIP:' ~/sbatch-staging/fliprefit.log | sed -n ${N}p; grep -c 'refit 3 ended' ~/sbatch-staging/fliprefit.log" 2>/dev/null | grep -v -E "^tput:|Pseudo-terminal")
  Z=$(echo "$L" | grep -o "nonsubmissions/[^ ]*\.zip" | head -1)
  if [ -n "$Z" ]; then
    mkdir -p "$REPO/$(dirname $Z)"
    scp -i $K -o BatchMode=yes "${H}:scratchDirectory/sar-benchmark/$Z" "$REPO/$Z" && echo "ZIP_FETCHED $REPO/$Z" && ls -l "$REPO/$Z"; exit 0
  elif echo "$L" | grep -q "tail ZIP: none"; then echo "FAILED $L"; exit 1
  elif [ "$(echo "$L" | tail -1)" != "0" ] && [ -n "$(echo "$L" | tail -1)" ]; then echo "REFIT3_FAILED"; exit 1; fi
  sleep 300
done
echo TIMEOUT; exit 2
