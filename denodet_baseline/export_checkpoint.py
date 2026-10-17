"""Export a stable DenoDet best checkpoint while its training job continues."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/GrokSAR'))
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/course-data')
    parser.add_argument('--converted', type=Path, default=ROOT / 'data/yolo11s-sar-512')
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / 'checkpoint.pth'
    # The running trainer replaces best checkpoints. Copy only a completed file,
    # then load the private copy; never modify training's files or configuration.
    for attempt in range(30):
        try:
            available = list(args.source.glob('best_*.pth'))
            available = [path for path in available if time.time() - path.stat().st_mtime > 10]
            if len(available) == 1:
                shutil.copy2(available[0], snapshot)
                chosen = available[0]
                break
        except FileNotFoundError:
            pass
        time.sleep(2)
    else:
        raise RuntimeError('No stable best checkpoint was available to snapshot')
    import torch
    import groksar.models
    import denodet_baseline.model
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmdet.registry import MODELS
    from denodet_baseline.export import export_run

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Export must run inside the allocated GPU job')
    cfg = Config.fromfile(str(args.source / 'config.py'))
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    model = MODELS.build(cfg.model)
    checkpoint = load_checkpoint(model, str(snapshot), map_location='cpu', strict=True)
    model.cuda().eval()
    settings = json.loads((args.source / 'benchmark_settings.json').read_text())
    settings.update(snapshot_source=str(chosen), snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                    snapshot_while_training=True, export_batch=args.batch,
                    checkpoint_epoch=checkpoint.get('meta', {}).get('epoch'))
    (args.output / 'snapshot.json').write_text(json.dumps(settings, indent=2) + '\n')
    dataset = args.source / 'dataset'
    metadata = json.loads((dataset / 'manifest.json').read_text())
    export_run(model, args, metadata, dataset, snapshot, settings)


if __name__ == '__main__':
    main()
