#!/bin/bash
# s1adfs-b11 (339126, Blackwell) is held so that when 338843 frees a slot, the L40S chain (339125) is the only job
# that can take it. Release b11 once the chain has started, or once flipsoup5.sh has moved the chain to Blackwell.
LOG=~/sbatch-staging/flipsoup3.log
until squeue -j 339125 -h -t R 2>/dev/null | grep -q . || grep -q "on a Blackwell slice: job" $LOG; do sleep 120; done
scontrol release 339126 && echo "$(date '+%m-%d %H:%M') s1adfs-b11 339126 released (chain started or moved)" >> $LOG
