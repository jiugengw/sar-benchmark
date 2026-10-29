#!/usr/bin/env python3
"""Class probabilities of a saved branch classifier (branch_head_study.py) for a set of images.

Same input as the submission writer: grayscale, resized to 512 with bilinear, copied into 3 channels; the frozen
detector runs up to the branch's split layer and the branch's own copy of the remaining backbone layers does the rest.

    python models/yolo/branch_head_predict.py --detector <detector.pt> --branch runs/<run>/branch_head.pt \\
        --images data/course-data/val/images.json --split-root data/course-data/val --out runs/<run>/val_probs.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--branch', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True, help='COCO json whose "images" are classified')
    parser.add_argument('--split-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    from PIL import Image
    from branch_head_study import BranchHead
    from input_filters import active_filter, apply as apply_input_filter
    from detector_support import load_detector

    device = torch.device(args.device)
    payload = torch.load(args.branch, map_location='cpu', weights_only=False)
    detector = load_detector(args.detector, 'yolo').model.float().to(device).eval()
    split, end = payload['split'], len(detector.yaml['backbone'])
    # A TerraMind-fusion detector: the branch runs the plain detector layers, as in branch_head_study.py
    from terramind_fusion import unwrap
    frozen = [unwrap(layer) for layer in detector.model[:split]]
    trunk = nn.ModuleList(copy.deepcopy(unwrap(detector.model[i])) for i in range(split, end))
    with torch.no_grad():
        x = torch.zeros(1, 3, 512, 512, device=device)
        for layer in frozen:
            x = layer(x)
        shapes = []
        for layer in trunk:
            x = layer(x)
            shapes.append(x.shape[1])
    head = BranchHead(trunk, [shapes[len(trunk) - 5], shapes[-1]]).to(device).eval()
    head.load_state_dict(payload['state_dict'])
    images = json.loads(args.images.read_text())['images']
    input_filter = active_filter()
    out = {}
    with torch.no_grad():
        for start in range(0, len(images), 16):
            group = images[start:start + 16]
            batch = []
            for im in group:
                with Image.open(args.split_root / im['file_name']) as original:
                    gray = np.asarray(original.convert('L').resize((512, 512), Image.Resampling.BILINEAR), np.uint8)
                # the detector's fixed input filter (SAR_INPUT_FILTER); 'none' copies the image into all 3 channels
                batch.append(apply_input_filter(input_filter, gray).transpose(2, 0, 1).astype(np.float32) / 255.0)
            x = torch.from_numpy(np.stack(batch)).to(device)
            for layer in frozen:
                x = layer(x)
            probs = torch.softmax(head(x), 1).cpu().tolist()
            for im, p in zip(group, probs):
                out[str(im['id'])] = p
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f'BRANCH_PROBS {len(out)} images -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
