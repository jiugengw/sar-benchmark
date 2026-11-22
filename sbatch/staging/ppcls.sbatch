#!/bin/bash
# Last cheap checks (29 Oct): image class from the heads plus the detector's boxes, and a
# re-fit of the post-processing (box-edge nudge, per-class hedge tiers) on 340514; CPU only. Results: ppcls_eval.out
#SBATCH --job-name=sar-ppcls-study
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --prefer=l40s
#SBATCH --time=03:00:00
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONHASHSEED=0 MPLCONFIGDIR=/tmp/sar-ppcls-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
{ echo "== $(date '+%m-%d %H:%M') heads + boxes, and post-processing re-fit, on 340514 (job $SLURM_JOB_ID)"
  python -u ~/sbatch-staging/pp_cls_study.py 2>&1 | grep -v -i warn; } >> ~/sbatch-staging/ppcls_eval.out
