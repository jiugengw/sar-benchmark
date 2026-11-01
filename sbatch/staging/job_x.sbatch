#!/bin/bash
# job_x: retrain only the detection head (layer 29 of 30, 1.4M of 21M weights) with the backbone and neck frozen, so
# the two-head image classifier's inputs and labels stay exactly the same (1 Nov, user: "think of how u can modify
# the detector slightly without retraining it all", then "Start it"). Bridge and playground images shown 3 times, their
# positive targets weighted x2; class-selective dataset flips, no augmentation, beta 8, as the final recipe.
# Ultralytics saves last.pt every epoch, so a run stopped at the deadline cut-off still leaves a usable head.
#   V=twin sbatch --job-name=job6 --export=ALL ~/sbatch-staging/job_x.sbatch   training split, from 344678 (holdout check)
#   V=all  sbatch --job-name=job7 --export=ALL ~/sbatch-staging/job_x.sbatch   all data, from the final flip soup
#   V=all2 sbatch --job-name=job11 --export=ALL --constraint=l40s ~/sbatch-staging/job_x.sbatch   the same with LR 0.005
#          (2.5x) over 10 epochs (8 on an A40, 5 on anything slower, to finish before the deadline), saved every epoch
#          (user, 1 Nov ~19:40: "use one gpu for more epochs ... start from differnt lr? and diff decay")
#SBATCH --job-name=job-x
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x-%j.out
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/job-x-$SLURM_JOB_ID PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 SAR_INPUT_FILTER=none
echo "start $(date '+%m-%d %H:%M') on $(nvidia-smi --query-gpu=name --format=csv,noheader); variant ${V:?set V}"
S="--no-augment --stem-stride 1 --tal-beta 8 --batch 16 --skip-evaluation --freeze 29
   --dataset-flips --dataset-flips-classes airport,bridge,harbor,oil_tank,playground,wind_turbine
   --repeat-classes bridge,playground --repeat-factor 3 --class-weights bridge:2,playground:2"
ALL="--all-data --source runs/soup-s1fs/alldata-s1fs-six.pt --classifier runs/memorise-lr2-332326-package/image_classifier.pt"
case $V in
  twin) python -u models/yolo/finetune_detector.py --tag ft40-x-twin-$SLURM_JOB_ID --lr0 0.002 --epochs 5 $S \
          --source runs/ft40-s1r-flipsel-344678/train/weights/best.pt --classifier runs/third-stage-331789-package/image_classifier.pt ;;
  all)  python -u models/yolo/finetune_detector.py --tag ft40-x-all-$SLURM_JOB_ID --lr0 0.002 --epochs 5 $S $ALL ;;
  all2) case $(nvidia-smi --query-gpu=name --format=csv,noheader) in *L40S*) E=10 ;; *A40*) E=8 ;; *) E=5 ;; esac
        echo "epochs $E"
        python -u models/yolo/finetune_detector.py --tag ft40-x-all2-$SLURM_JOB_ID --lr0 0.005 --epochs $E \
          --save-period 1 $S $ALL ;;
  *) echo "unknown variant $V"; exit 2 ;;
esac
echo "JOB_X_DONE $V $(date '+%m-%d %H:%M')"
