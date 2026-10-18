"""Fine-tune Dome-DETR-M (AI-TOD optical weights) on our SAR training split at 512 x 512, then export the holdout.

Dome-DETR (ACM MM 2025) is a D-FINE detector for tiny objects: a density map picks where to put queries, so crowded
images get more of them (300 to 1500). 24.4M parameters, under the 40M trainable limit. Pretraining: HGNetV2-B2 on
ImageNet, then the whole detector on AI-TOD v2 (optical aerial photos). No SAR-pretrained weights.

Changes from the authors' AI-TOD recipe: 512 x 512 input instead of 800 (every image stretched to 512, as in our YOLO
runs; the TA confirmed anything up to 512 is allowed), 9 of our classes in the AI-TOD class slots, vertical flips added (SAR scenes have no up), colour jitter dropped (grayscale images),
fewer epochs because we start from a trained detector. The last `--clean-epochs` run without zoom/crop augmentation,
as Dome and D-FINE do. The best epoch by holdout mAP is kept, as for YOLO's best.pt.

    python models/dome/train.py --tag dome-m-test --epochs 48 --batch 8
"""
from __future__ import annotations

import argparse
import json
import runpy
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.dome.common import CHECKPOINT, DATA, DOME_ROOT, ROOT, use_dome  # noqa: E402
from models.dome.prepare_data import prepare  # noqa: E402


def write_config(path: Path, data: Path, output: Path, epochs: int, clean_epochs: int, batch: int, size: int) -> None:
    """Our overrides on top of the authors' Dome-M AI-TOD config (lists replace, dictionaries merge)."""
    import yaml

    stop = epochs - clean_epochs
    config = {
        '__include__': [str(DOME_ROOT / 'configs/dome/Dome-M-AITOD.yml')],
        'output_dir': str(output),
        'num_classes': 9,
        'remap_mscoco_category': False,
        'eval_spatial_size': [size, size],
        'evaluator': {'type': 'CocoEvaluator', 'iou_types': ['bbox']},
        'HGNetv2': {'pretrained': False},  # every weight comes from the AI-TOD checkpoint
        'epoches': epochs,
        'checkpoint_freq': 1000,
        'lr_scheduler': {'type': 'MultiStepLR', 'milestones': [max(1, stop - 8)], 'gamma': 0.5},
        'train_dataloader': {
            'dataset': {
                'img_folder': str(DATA / 'train'), 'ann_file': str(data / 'train.json'),
                'transforms': {
                    'ops': [
                        {'type': 'RandomZoomOut', 'fill': 0},
                        {'type': 'RandomIoUCrop', 'p': 0.8},
                        {'type': 'SanitizeBoundingBoxes', 'min_size': 1},
                        {'type': 'RandomHorizontalFlip'},
                        {'type': 'RandomVerticalFlip'},
                        {'type': 'Resize', 'size': [size, size]},
                        {'type': 'SanitizeBoundingBoxes', 'min_size': 1},
                        {'type': 'ConvertPILImage', 'dtype': 'float32', 'scale': True},
                        {'type': 'ConvertBoxes', 'fmt': 'cxcywh', 'normalize': True},
                    ],
                    'policy': {'name': 'stop_epoch', 'epoch': stop, 'ops': ['RandomZoomOut', 'RandomIoUCrop']},
                },
            },
            # base_size sets the multi-scale training sizes (about 0.75x to 1.25x); prediction is always at `size`
            'collate_fn': {'base_size': size, 'stop_epoch': stop, 'ema_restart_decay': 0.9999},
            'total_batch_size': batch,
            'num_workers': 4,
        },
        'val_dataloader': {
            'dataset': {
                'img_folder': str(DATA / 'train'), 'ann_file': str(data / 'holdout.json'),
                'transforms': {'ops': [{'type': 'Resize', 'size': [size, size]},
                                       {'type': 'ConvertPILImage', 'dtype': 'float32', 'scale': True}]},
            },
            'total_batch_size': 16,
            'num_workers': 4,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True, help='runs/<tag>, results/<tag>')
    parser.add_argument('--epochs', type=int, default=48)
    parser.add_argument('--clean-epochs', type=int, default=8, help='final epochs without zoom-out / crop')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT, help='weights to fine-tune from')
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'{args.checkpoint}; run scripts/setup_dome.sh first')
    if args.size > 512:
        raise ValueError('the competition allows at most 512 x 512 input')

    run = ROOT / 'runs' / args.tag
    data = run / 'data'
    prepare(data)
    config = run / 'config.yml'
    write_config(config, data, run / 'train', args.epochs, args.clean_epochs, args.batch, args.size)
    (run / 'experiment.json').write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                     'model': 'Dome-DETR-M', 'pretrained': 'AI-TOD v2 (optical)'},
                                                    indent=2) + '\n')

    use_dome()
    import torchvision.transforms.v2 as T
    from src.core import register
    register()(T.RandomVerticalFlip)  # Dome registers only the horizontal flip

    started = time.monotonic()
    sys.argv = ['train.py', '-c', str(config), '-t', str(args.checkpoint), '--seed', str(args.seed), '--use-amp',
                '--output-dir', str(run / 'train')]
    runpy.run_path(str(DOME_ROOT / 'train.py'), run_name='__main__')
    (run / 'compute.json').write_text(json.dumps({'wall_seconds': time.monotonic() - started}) + '\n')

    curves = ROOT / 'results/training-curves' / args.tag  # runs/ is not committed; keep the per-epoch log
    curves.mkdir(parents=True, exist_ok=True)
    for name in ('log.txt',):
        if (run / 'train' / name).is_file():
            shutil.copy2(run / 'train' / name, curves / name)
    shutil.copy2(config, curves / 'config.yml')
    shutil.copy2(run / 'experiment.json', curves / 'experiment.json')


if __name__ == '__main__':
    main()
