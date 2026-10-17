#!/bin/bash
# Official DenoDet V2 FFTResNet-50 + GFL, with a shared image classification head.
# ImageNet backbone initialization only; no SAR-pretrained model weights.
# Submit from the project root: sbatch sbatch/denodet_v2.sbatch
#SBATCH --job-name=sar-denodet-v2
#SBATCH --partition=student
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0 CUDA/12.8.0
source .denodet-venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MAX_JOBS=2
export CUDA_HOME=/opt/apps/software/CUDA/12.8.0
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
export MMCV_WITH_OPS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}/third_party/GrokSAR${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/sar-denodet-${SLURM_JOB_ID}"

OUTPUT="runs/denodet-v2-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT"
python denodet_baseline/patch_mmcv.py
if ! python -c 'import mmcv._ext'; then
    echo "Building MMCV CUDA operators for Blackwell (first run only)"
    python -m pip install --no-deps --no-build-isolation ./third_party/mmcv > "$OUTPUT/mmcv-build.log" 2>&1 || {
        tail -100 "$OUTPUT/mmcv-build.log"
        exit 1
    }
fi

python -u denodet_baseline/train.py \
    --output "$OUTPUT" --epochs "${EPOCHS:-48}" --batch "${BATCH:-8}" --workers 4
echo "DenoDet V2 finished: $OUTPUT"
