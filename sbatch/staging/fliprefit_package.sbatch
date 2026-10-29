#!/bin/bash
# Package the flip-refit soup (DET, RUN from the environment; fliprefit.sh), rounded + hedged.
#SBATCH --job-name=sar-s1fsr-pkg
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a40|rtx_pro_6000_blackwell_max
#SBATCH --prefer=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -euo pipefail
TAG=${TAG:-alldata-s1fsr}
cd ~/scratchDirectory/sar-benchmark
source /opt/apps/lmod/lmod/init/bash; module load Python/3.12.8-GCCcore-13.3.0; source .venv/bin/activate
export PYTHONPATH=$PWD:$PWD/models/yolo MPLCONFIGDIR=/tmp/s1fsr-$SLURM_JOB_ID PYTHONUNBUFFERED=1
python -u models/yolo/package_candidate.py --tag $TAG --detector $DET --curves $RUN/train/results.csv --box-audit results/memorise-lr2-332326/box_bias_crossfit.json --classifier-all-data
Z=$(ls -t nonsubmissions/*/$TAG/$TAG.zip | head -1); D=$(dirname $Z)
python models/yolo/round_boxes.py zip $Z $D/$TAG-rounded.zip --images data/course-data/val/images.json
python models/yolo/hedge_study.py zip $D/$TAG-rounded.zip $D/$TAG-rounded-hedged.zip
echo "S1FSR_ZIP_READY $D/$TAG-rounded-hedged.zip"
# Low-score tail (28 Oct: Codabench 59.93 -> 60.04 on refit 2): the same detector at conf 1e-5, its sub-0.001 boxes
# ranked below every box (models/yolo/lowconf_tail.py). Keeps lowconf.zip so the two-head ZIP can get the tail on CPU.
{ python -u models/yolo/make_submission.py --weights $DET --detection-branch one2many --out $D/$TAG-lowconf.zip --conf 1e-5 --max-det 1000 \
  && python models/yolo/lowconf_tail.py $D/$TAG-rounded-hedged.zip $D/$TAG-lowconf.zip $D/$TAG-rounded-hedged-tail.zip \
  && echo "S1FSR_TAIL_READY $D/$TAG-rounded-hedged-tail.zip"; } || echo "S1FSR_TAIL_FAILED"
