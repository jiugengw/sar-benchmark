#!/bin/bash
# All-data refit with frozen TerraMind features fused into the detector (29 Oct): refit 2 (340754, Codabench 60.04 with the tail) + 10 epochs of the same class-selective flip memorise, with
# terramind_fusion.py (frozen TerraMind-1.0-base tokens added at backbone layers 6 and 10 through zero-initialised 1x1
# convolutions; one network). No training-split test first, by the user's choice: Codabench decides. Driver:
#   cd ~/scratchDirectory/sar-benchmark
#   J=$(sbatch --parsable -o runs/%x-%j.out ~/sbatch-staging/fliprefitT.sbatch)
#   nohup bash ~/sbatch-staging/refit_ship.sh $J T > /dev/null 2>&1 &
#SBATCH --job-name=sar-ft40-fliprefitT
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --time=16:00:00
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-frT-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u models/yolo/finetune_detector.py --tag ft40-s1adfsrT-$SLURM_JOB_ID \
    --source runs/ft40-s1adfsr2-340754/train/weights/last.pt \
    --classifier runs/memorise-lr2-332326-package/image_classifier.pt --all-data --skip-evaluation --no-augment \
    --lr0 0.001 --epochs 10 --stem-stride 1 --batch 16 --tal-beta 8 --dataset-flips \
    --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine --terramind-fusion \
    && echo FLIPREFITT_ALLDATA_DONE
