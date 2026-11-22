#!/bin/bash
# 29 Oct: does Gaussian input smoothing help only the noisiest images? 340514 vs 342730 by sharpness.
# CPU only, analysis only.
#SBATCH --job-name=sar-sharp-split
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --prefer=l40s
#SBATCH --time=02:00:00
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONHASHSEED=0 MPLCONFIGDIR=/tmp/sar-ss-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
{ echo "== $(date '+%m-%d %H:%M') smoothing by image sharpness, 340514 vs 342730 (job $SLURM_JOB_ID)"
  python -u ~/sbatch-staging/sharp_split.py ft40-s1r-fliprefit-340514 ft40-s1r-gauss1-342730 2>&1 | grep -v -i warn; } >> ~/sbatch-staging/sharp_split.out
