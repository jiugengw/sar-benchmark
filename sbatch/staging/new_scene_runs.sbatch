#!/bin/bash
# 29 Oct: rank existing training-split runs by new-scene AP per class.
# CPU only, analysis only (private-m4x* rows are diagnostic, never submittable). Results: new_scene_runs.out
#SBATCH --job-name=sar-new-scene-runs
#SBATCH --partition=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --prefer=l40s
#SBATCH --time=03:00:00
set -uo pipefail
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONHASHSEED=0 MPLCONFIGDIR=/tmp/sar-nsr-$SLURM_JOB_ID
export PYTHONPATH=$PWD:$PWD/models/yolo
RUNS="cmp-normal-100 cmp-normal-plain third-stage-331789 ft40-memorise-lr2-332326 ft40-memorise-photo-332877 ft40-memorise-flips-332161
 ft40-boxpaste-memorise-333330 ft40-m4repeat-memorise-333334 ft40-water-memorise-332236 ft40-l-memorise-332235 ft40-cw3-memorise-337406 ft40-hard-memorise-337459
 cmp-s1-100 cmp-s1-plain cmp-s1-native cmp-s1140-auto cmp-l-s1-100 ft40-s1r-memorise-337666 ft40-s1r-flipmem-338727 ft40-s1r-flipspk-338730
 ft40-s1r-flipanneal-338803 ft40-s1r-fliprefit-340514 ft40-s1r-twinjit-341749 ft40-s1r-gauss1-342730 ft40-s1r-bilateral-342729 soup-s1r-flip-spk
 private-m4x0-control-third-333334 private-m4x10000-third-334409 private-m4x30000-third-334410 private-m4x103886-third-334411"
{ echo "== $(date '+%m-%d %H:%M') new-scene AP per class by run (job $SLURM_JOB_ID)"
  python -u ~/sbatch-staging/new_scene_runs.py $RUNS 2>&1 | grep -v -i warn; } >> ~/sbatch-staging/new_scene_runs.out
