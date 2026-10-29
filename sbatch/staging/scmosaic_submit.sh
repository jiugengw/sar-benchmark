#!/bin/bash
# queue the same-class mosaic run (bp4 recipe, mosaic 0.5, then memorise beta 8, package) once a submit slot is free
cd ~/scratchDirectory/sar-benchmark
while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done
J=$(MEMORISE_EXTRA="--tal-beta 8" PACKAGE=1 sbatch --parsable --time=08:00:00 sbatch/third_then_memorise.sbatch bp4-scmosaic --mosaic 0.5 --same-class-mosaic --repeat-classes airport,harbor,oil_tank,wind_turbine,bridge:4,playground:4 --repeat-factor 2 --repeat-width 512)
echo "$(date +%H:%M) same-class mosaic submitted: $J" >> ~/sbatch-staging/s1_chain.log
