#!/bin/bash
# Jitter soup ingredients back to back in ONE job (the GPU is not released between them): the class-selective flip
# memorise recipe of the +65.55 soup plus brightness (+-20%, --photometric) and sharpness jitter (--sharpness-prob 0.5),
# so the model also recognises twins (training scenes re-processed brighter or blurrier). LIST="name:beta ..." from
# the environment, e.g. LIST="s1adfj-b8:8 s1adfj-b10:10" sbatch --export=ALL flipjit_chain.sbatch
#SBATCH --job-name=sar-ft40-jchain
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-jchain-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
T=runs/ft40-s1ad-third-338033/train/weights/last.pt; CLS=runs/memorise-lr2-332326-package/image_classifier.pt
S="--source $T --classifier $CLS --all-data --skip-evaluation --no-augment --lr0 0.002 --epochs 20 --stem-stride 1 --batch 16 --dataset-flips --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine --photometric --sharpness-prob 0.5"
left() {  # seconds before this job's time limit
  python3 -c "import sys; d, _, r = sys.argv[1].rpartition('-'); p = [int(x) for x in r.split(':')]; p = [0] * (3 - len(p)) + p
print((int(d) if d else 0) * 86400 + p[0] * 3600 + p[1] * 60 + p[2])" "$(squeue -h -j $SLURM_JOB_ID -o %L)"
}
last=0
for spec in $LIST; do
  name=${spec%%:*}; b=${spec#*:}
  # skip an ingredient that would not finish before the time limit (the driver resubmits it alone)
  if [ $last -gt 0 ] && [ $(left) -lt $((last * 11 / 10 + 900)) ]; then echo "CHAIN skip $name: $(left) s left, the last one took $last s"; continue; fi
  echo "CHAIN start $name (beta $b) $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  t0=$(date +%s)
  python -u models/yolo/finetune_detector.py --tag "ft40-${name}-${SLURM_JOB_ID}" $S --tal-beta $b || echo "CHAIN $name FAILED"
  last=$(( $(date +%s) - t0 ))
  echo "CHAIN end $name $(date '+%m-%d %H:%M')"
done
echo CHAIN_DONE
