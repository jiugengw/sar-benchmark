#!/bin/bash
# One change to the third stage, then the control's memorise, in one job (one queue slot): the best recipe's last two
# stages, so the result compares directly with memorise lr 0.002 (332326, clean 50.40, official-mix 51.80).
#   third stage: 40 epochs from the DOTA chain's speckle stage, lr 0.001, the usual jitter + the options given here
#   memorise:    40 epochs without augmentation, lr 0.002
#   sbatch sbatch/third_then_memorise.sbatch boxpaste --box-paste 0.5
#SBATCH --job-name=sar-3rd-mem
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/sar-3rd-mem-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

NAME="${1:?Give a short experiment name}"
shift
THIRD="ft40-${NAME}-third-${SLURM_JOB_ID}"
python -u models/yolo/finetune_detector.py --tag "$THIRD" \
    --source runs/chain-dota-330938-speckle/train/weights/best.pt \
    --classifier runs/chain-dota-330938-package/image_classifier.pt --skip-evaluation "$@"
python -u models/yolo/finetune_detector.py --tag "ft40-${NAME}-memorise-${SLURM_JOB_ID}" \
    --source "runs/$THIRD/train/weights/best.pt" \
    --classifier runs/third-stage-331789-package/image_classifier.pt --no-augment --lr0 0.002
