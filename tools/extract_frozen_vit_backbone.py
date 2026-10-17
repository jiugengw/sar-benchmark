"""Recover the unchanged ImageNet-pretrained ViT weights from a frozen MoE-LoRA run."""
import argparse
from pathlib import Path

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root, for sarbench
from sarbench.backbones import build_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-checkpoint', type=Path, required=True,
                        help='model.pt from a run where the original ViT backbone was frozen')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    trained = torch.load(args.run_checkpoint, map_location='cpu', weights_only=True)
    prefix = 'backbone.vit.'
    state = {
        key[len(prefix):].replace('.base.', '.'): value
        for key, value in trained.items()
        if key.startswith(prefix) and not any(part in key for part in ('.down.', '.up.', '.routers.'))
    }
    if not state:
        raise ValueError(f'No ViT backbone weights found in {args.run_checkpoint}')

    # Strict loading confirms this recovered state matches the 512 px, one-channel ViT used by the trainer.
    model = build_backbone('vit', pretrained=False)
    model.vit.load_state_dict(state, strict=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    print(f'Saved {len(state)} unchanged backbone tensors ({sum(t.numel() for t in state.values()):,} parameters) to {args.out}')


if __name__ == '__main__':
    main()
