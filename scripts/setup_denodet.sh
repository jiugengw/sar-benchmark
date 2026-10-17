#!/bin/bash
# Prepare an isolated OpenMMLab environment; the sbatch job builds CUDA ops.
set -euo pipefail
cd "$(dirname "$0")/.."
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0

mkdir -p third_party pretrained runs
if [[ ! -d third_party/GrokSAR/.git ]]; then
    git clone https://github.com/GrokCV/GrokSAR.git third_party/GrokSAR
fi
git -C third_party/GrokSAR checkout --detach d41fb0e5cbaaf9121174357c7a48547b278fe458
if [[ ! -d third_party/mmcv/.git ]]; then
    git clone --depth 1 --branch v2.0.1 https://github.com/open-mmlab/mmcv.git third_party/mmcv
fi
if [[ "$(git -C third_party/mmcv rev-parse HEAD)" != fc038a386a42d1f36d9ec49169d5817a52c70b8e ]]; then
    echo "Unexpected MMCV revision; expected the pinned v2.0.1 sources" >&2
    exit 1
fi
if [[ ! -x .denodet-venv/bin/python ]]; then
    .venv/bin/python -m venv .denodet-venv
fi
.denodet-venv/bin/python - <<'PY'
from pathlib import Path
import sys
packages = Path(sys.prefix) / 'lib/python3.12/site-packages'
(packages / 'sar_shared.pth').write_text(str(Path('.venv/lib/python3.12/site-packages').resolve()) + '\n')
PY
# Install into this environment only. Torch/CUDA stay in the existing .venv.
.denodet-venv/bin/python -m pip install --no-deps -r requirements-denodet.txt
if [[ ! -s pretrained/resnet50-0676ba61.pth ]]; then
    curl --fail --location --output pretrained/resnet50-0676ba61.pth.partial \
        https://download.pytorch.org/models/resnet50-0676ba61.pth
    mv pretrained/resnet50-0676ba61.pth.partial pretrained/resnet50-0676ba61.pth
fi
.denodet-venv/bin/python denodet_baseline/patch_mmcv.py
echo 'Ready: sbatch sbatch/denodet_v2.sbatch'
