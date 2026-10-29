#!/bin/bash
cd ~/scratchDirectory/sar-benchmark
until ls nonsubmissions/*/alldata-soup6/alldata-soup6-rounded-hedged.zip >/dev/null 2>&1; do sleep 120; done
Z=$(ls -t nonsubmissions/*/alldata-soup6/alldata-soup6-rounded-hedged.zip | head -1)
M=runs/alldata-soup6-package/alldata-soup6_base.probabilities.json
[ -f $M ] || M=$(ls -t runs/alldata-soup6-package/*probabilities.json | head -1)
while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done
echo "$(date '+%m-%d %H:%M') two-head job: $(sbatch --parsable /tmp/twohead.sbatch runs/soup-bp4/alldata-six.pt alldata-six $Z $M) ($Z, $M)" >> ~/sbatch-staging/alldata_soup.log
