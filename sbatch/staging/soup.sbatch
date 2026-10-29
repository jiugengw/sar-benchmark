#!/bin/bash
#SBATCH --job-name=sar-soup
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=runs/%x-%j.out
set -euo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/soup-$SLURM_JOB_ID PYTHONUNBUFFERED=1
python -u models/yolo/soup_study.py --tag soup-bp4 --soup b8b12=$B8,$B12 --soup b8spk3=$B8,$S5,$S8 --soup five=$B8,$B12,$SA,$S5,$S8 --soup six=$B8,$B12,$SA,$S5,$S8,$PV
