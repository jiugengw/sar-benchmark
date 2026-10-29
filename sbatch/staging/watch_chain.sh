#!/bin/bash
# Poll the server every 10 min (tolerating SSH drops) until the chain finishes or the driver dies.
while true; do
  out=$(ssh -o ConnectTimeout=20 aetherius-wsl 'cd ~/sar-live; grep "===" ls1_chain.log; pgrep -f run_ls1_chain.sh >/dev/null && echo DRIVER_ALIVE || echo DRIVER_DEAD' 2>/dev/null) || { sleep 600; continue; }
  if echo "$out" | grep -q "chain finished" || echo "$out" | grep -q DRIVER_DEAD; then echo "$out"; exit 0; fi
  sleep 600
done
