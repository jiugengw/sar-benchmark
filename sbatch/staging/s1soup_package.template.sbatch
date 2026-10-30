#!/bin/bash
#SBATCH --job-name=sar-s1soup-pkg
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/s1soup-$SLURM_JOB_ID PYTHONUNBUFFERED=1
python -u models/yolo/package_candidate.py --tag alldata-s1-soup6 --detector runs/soup-s1/alldata-s1-six.pt --curves runs/ft40-s1ad-memorise-338116/train/results.csv --box-audit results/memorise-lr2-332326/box_bias_crossfit.json --classifier-all-data
Z=$(ls -t nonsubmissions/*/alldata-s1-soup6/alldata-s1-soup6.zip | head -1); D=$(dirname $Z)
python models/yolo/round_boxes.py zip $Z $D/alldata-s1-soup6-rounded.zip --images data/course-data/val/images.json
python models/yolo/hedge_study.py zip $D/alldata-s1-soup6-rounded.zip $D/alldata-s1-soup6-rounded-hedged.zip
echo "S1_SOUP_ZIP_READY $D/alldata-s1-soup6-rounded-hedged.zip"
