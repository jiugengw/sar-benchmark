"""Shared paths and class mapping for the Dome-DETR experiment (third_party/Dome-DETR, see scripts/setup_dome.sh)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOME_ROOT = ROOT / 'third_party/Dome-DETR'
DOME_COMMIT = '2dde3bc1946a3e9fad9abd0612b59fc39bd6b861'
# Dome-M trained on AI-TOD v2 (optical aerial photos, ImageNet HGNetV2 backbone); no SAR data.
CHECKPOINT = ROOT / 'pretrained/dome/aitod-m-best.pth'
DATA = ROOT / 'data/course-data'
SPLIT = ROOT / 'results/yolo26m-100-p2-328339/seed_0/split.json'

# Our category id -> Dome label slot. The AI-TOD checkpoint has 9 class slots: 0 unused, then airplane, bridge,
# storage-tank, ship, swimming-pool, vehicle, person, wind-mill (the AI-TOD class order). Putting each of our classes
# in the slot of its AI-TOD counterpart lets every class head load and start from a related class; airport, harbor
# and playground take the slots of classes we do not have. If the order guess were wrong, nothing breaks: the heads
# would simply start from unrelated classes, as a freshly initialised head would.
CATEGORY_TO_SLOT = {1: 1,  # aircraft -> airplane
                    3: 2,  # bridge -> bridge
                    6: 3,  # oil_tank -> storage-tank
                    8: 4,  # ship -> ship
                    7: 5,  # playground -> swimming-pool
                    4: 6,  # car -> vehicle
                    5: 7,  # harbor -> person (unrelated)
                    9: 8,  # wind_turbine -> wind-mill
                    2: 0}  # airport -> unused slot
SLOT_TO_CATEGORY = {slot: category for category, slot in CATEGORY_TO_SLOT.items()}


def use_dome() -> None:
    """Make Dome-DETR's `src` and `tools` packages importable, ahead of this repository's own tools/ folder.

    Dome imports the AI-TOD evaluator package at start-up even when it is not used; it only ships a Python 3.9 wheel,
    so when it is missing its name points at faster_coco_eval (we evaluate with the standard COCO evaluator)."""
    if not (DOME_ROOT / 'train.py').is_file():
        raise FileNotFoundError(f'{DOME_ROOT} is missing; run scripts/setup_dome.sh first')
    sys.path.insert(0, str(DOME_ROOT))
    try:
        import faster_coco_eval_aitod  # noqa: F401
    except ImportError:
        import faster_coco_eval
        import faster_coco_eval.core
        import faster_coco_eval.core.mask
        sys.modules['faster_coco_eval_aitod'] = faster_coco_eval
        sys.modules['faster_coco_eval_aitod.core'] = faster_coco_eval.core
        sys.modules['faster_coco_eval_aitod.core.mask'] = faster_coco_eval.core.mask
