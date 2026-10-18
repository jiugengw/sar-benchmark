#!/bin/bash
# Prepare Dome-DETR on the login node (needs internet): pinned source, an isolated venv, the AI-TOD Dome-M weights.
# Torch/CUDA stay in the existing .venv; the new .dome-venv only adds Dome's extra packages.
set -euo pipefail
cd "$(dirname "$0")/.."
source /opt/apps/lmod/lmod/init/bash
module load Python/3.12.8-GCCcore-13.3.0

COMMIT=2dde3bc1946a3e9fad9abd0612b59fc39bd6b861
mkdir -p third_party pretrained/dome runs
if [[ ! -d third_party/Dome-DETR/.git ]]; then
    git clone https://github.com/RicePasteM/Dome-DETR.git third_party/Dome-DETR
fi
git -C third_party/Dome-DETR checkout --detach "$COMMIT"

if [[ ! -x .dome-venv/bin/python ]]; then
    .venv/bin/python -m venv .dome-venv
fi
.dome-venv/bin/python - <<'PY'
from pathlib import Path
import sys
packages = Path(sys.prefix) / 'lib/python3.12/site-packages'
(packages / 'sar_shared.pth').write_text(str(Path('.venv/lib/python3.12/site-packages').resolve()) + '\n')
PY
# Keep the shared torch, torchvision, numpy and pillow exactly as they are.
.venv/bin/python -m pip freeze | grep -iE '^(torch|torchvision|numpy|pillow)==' > runs/dome-constraints.txt
.dome-venv/bin/python -m pip install -c runs/dome-constraints.txt \
    'faster-coco-eval>=1.6.5' PyYAML tensorboard scipy calflops transformers scikit-image matplotlib
# Only used by Dome's AI-TOD evaluator, which we do not use; it has no Python 3.12 wheel, so a failure is fine.
.dome-venv/bin/python -m pip install -c runs/dome-constraints.txt faster-coco-eval-aitod \
    || echo 'faster-coco-eval-aitod not installed (not needed: models/dome/common.py aliases it)'

CKPT=pretrained/dome/aitod-m-best.pth
if [[ ! -s "$CKPT" ]]; then
    curl --fail --location --output "$CKPT.partial" \
        https://huggingface.co/RicePasteM/Dome-DETR/resolve/main/best_ckpts_dome_2026/aitod-m-best.pth
    mv "$CKPT.partial" "$CKPT"
fi
echo "d0242732410ac329a441e2a39b484bbfbb5f7196170cc0379a761ac0cf835d6b  $CKPT" | sha256sum --check

PYTHONPATH="$PWD" .dome-venv/bin/python -c '
from models.dome.common import use_dome
use_dome()
import src.zoo.dome, src.data
print("Dome-DETR imports OK")'
echo 'Ready: sbatch sbatch/dome_detr.sbatch'
