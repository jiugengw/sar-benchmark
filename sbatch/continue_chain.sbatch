#!/bin/bash
# The speckle-cand recipe on top of a trained 100-epoch YOLO26-P2: +40 plain, +40 speckle, classifier, box nudge.
# Submit from the project directory with a name and the 100-epoch run folder, e.g.:
#   sbatch sbatch/continue_chain.sbatch dota runs/yolo26m-p2-dota-verified-330705
# Add a third argument "plain" to skip the speckle stage. Anything after it is passed on, e.g. the all-data version:
#   sbatch sbatch/continue_chain.sbatch dota-all runs/yolo26m-p2-dota-verified-330705 plain,speckle \
#       --all-data --box-audit results/chain-dota-330938/box_bias_crossfit.json
#SBATCH --job-name=sar-chain
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="/tmp/sar-chain-${SLURM_JOB_ID}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/models/yolo${PYTHONPATH:+:${PYTHONPATH}}"

NAME="${1:?Give a short name, e.g. dota}"
RUN="${2:?Give the 100-epoch run folder, e.g. runs/yolo26m-p2-dota-verified-330705}"
STAGES="${3:-plain,speckle}"
shift $(( $# < 3 ? $# : 3 ))
python -u models/yolo/continue_chain.py --tag "chain-${NAME}-${SLURM_JOB_ID}" \
    --source "$RUN/train/weights/best.pt" --classifier "$RUN/image_classifier.pt" --stages "$STAGES" "$@"
