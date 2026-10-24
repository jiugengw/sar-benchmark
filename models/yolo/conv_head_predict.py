#!/usr/bin/env python3
"""Class probabilities of a saved convolutional classification branch (conv_head_study.py) for a set of images.

Same input as the submission writer: grayscale, resized to 512 with bilinear, copied into 3 channels; the detector
(frozen) supplies the feature maps the branch reads.

    python models/yolo/conv_head_predict.py --detector runs/<soup>.pt --branch runs/<conv>/conv_head.pt \\
        --images data/course-data/val/images.json --split-root data/course-data/val --out runs/<conv>/val_probs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

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
    from conv_head_study import ConvBranch
    from detector_support import detection_head, feature_maps_from_hook, load_detector

    device = torch.device(args.device)
    payload = torch.load(args.branch, map_location='cpu', weights_only=False)
    branch = ConvBranch(payload['channels'], width=payload['width']).to(device).eval()
    branch.load_state_dict(payload['state_dict'])
    model = load_detector(args.detector, 'yolo').model.to(device).eval()
    captured = []
    detection_head(model).register_forward_hook(
        lambda _m, inputs, output: captured.append([f.detach() for f in feature_maps_from_hook('yolo', inputs, output)]))
    images = json.loads(args.images.read_text())['images']
    out = {}
    with torch.no_grad():
        for start in range(0, len(images), 16):
            group = images[start:start + 16]
            batch = []
            for im in group:
                with Image.open(args.split_root / im['file_name']) as original:
                    gray = np.asarray(original.convert('L').resize((512, 512), Image.Resampling.BILINEAR), np.float32)
                batch.append(np.repeat(gray[None] / 255.0, 3, axis=0))
            captured.clear()
            model(torch.from_numpy(np.stack(batch)).to(device))
            probs = torch.softmax(branch(captured.pop()), 1).cpu().tolist()
            for im, p in zip(group, probs):
                out[str(im['id'])] = p
    args.out.write_text(json.dumps(out))
    print(f'CONV_PROBS {len(out)} images -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
