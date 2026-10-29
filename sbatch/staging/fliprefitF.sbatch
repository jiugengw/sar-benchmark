#!/bin/bash
# All-data refit with a fixed input filter (SAR_INPUT_FILTER from the environment: bilateral or gauss1; 29 Oct). Refit 2
# (340754, Codabench 60.04 with the tail) plus 10 epochs of the same class-selective flip memorise, every image (train,
# holdout, and later the test images in make_submission / the heads) smoothed by the same filter, so a twin and its
# training copy look more alike. Only launched if the holdout test (filter_test.sbatch) passed its rule. Driver:
#   cd ~/scratchDirectory/sar-benchmark; export SAR_INPUT_FILTER=<filter>
#   J=$(sbatch --parsable -o runs/%x-%j.out --export=ALL ~/sbatch-staging/fliprefitF.sbatch)
#   nohup bash ~/sbatch-staging/refit_ship.sh $J F > /dev/null 2>&1 &
#SBATCH --job-name=sar-ft40-fliprefitF
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --time=08:00:00
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/sar-frF-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
case "${SAR_INPUT_FILTER:-}" in bilateral|gauss1) ;; *) echo "SAR_INPUT_FILTER must be bilateral or gauss1"; exit 1;; esac
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader), input filter $SAR_INPUT_FILTER"
python -u models/yolo/finetune_detector.py --tag ft40-s1adfsrF-$SLURM_JOB_ID \
    --source runs/ft40-s1adfsr2-340754/train/weights/last.pt \
    --classifier runs/memorise-lr2-332326-package/image_classifier.pt --all-data --skip-evaluation --no-augment \
    --lr0 0.001 --epochs 10 --stem-stride 1 --batch 16 --tal-beta 8 --dataset-flips \
    --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine && echo FLIPREFITF_ALLDATA_DONE
