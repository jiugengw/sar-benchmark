"""Evaluate single-model confidence calibration locally, then package the selected variant."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from sarbench.adapters import add_adapters
from sarbench.backbones import build_backbone
from sarbench.data import SARMultiTask
from sarbench.model import MultiTaskModel
from train import evaluate, make_loader, validation_delta_m
from sarbench.reporting import ComputeTracker, provenance
from sarbench.twins import clean_image_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/course-data'))
    parser.add_argument('--holdout', type=Path, help='saved labeled holdout annotations, if outside the run')
    args = parser.parse_args()
    summary = json.loads((args.run / 'metrics.json').read_text())
    config = dict(summary['args'])
    config.setdefault('geometry', 'stretch')
    config.setdefault('class_sampling', 'uniform')
    config.setdefault('limit', None)
    config.update(data=args.data, photometric=False, workers=4)
    cfg = SimpleNamespace(**config)
    rows = SARMultiTask(args.data, limit=cfg.limit)
    permutation = torch.randperm(len(rows), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    indices = permutation[:max(1, round(len(rows) * 0.1))]
    loader = make_loader(cfg, indices, train=False)
    expected_ids = {rows.coco_images[rows.rows[i]['file_name']]['id'] for i in indices}
    gt_path = args.holdout or args.run / 'instances_val_holdout.json'
    gt = json.loads(gt_path.read_text())
    if expected_ids != {im['id'] for im in gt['images']}:
        raise ValueError('Saved holdout does not match the reconstructed validation split')
    with ComputeTracker(args.run / 'calibration_compute.json', 'calibration', provenance(config)):
        backbone = build_backbone(cfg.backbone, pretrained=False)
        if cfg.adapt != 'full':
            add_adapters(backbone, cfg.adapt)
        model = MultiTaskModel(backbone, task_routing=cfg.adapt == 'moelora',
                               anchor_profile=config.get('anchor_profile', 'default'))
        model.load_state_dict(torch.load(args.run / 'model.pt', map_location='cpu', weights_only=True))
        model.cuda().eval()
        results = []
        clean_ids = clean_image_ids(gt['images'])
        for power in (0.0, 0.5, 1.0):
            metrics, detections = evaluate(model, loader, gt_path, power, clean_ids=clean_ids)
            metrics['delta_m'] = validation_delta_m(metrics)
            metrics['clean']['delta_m'] = validation_delta_m(metrics['clean'])
            results.append({'class_score_power': power, **metrics})
            (args.run / f'predictions_calibration_{power}.json').write_text(json.dumps(detections))
            print(f"power={power}: F1={metrics['macro_f1']:.4f}, mAP={metrics['mAP']:.4f}, Delta m={metrics['delta_m']:+.2f}; "
                  f"clean ({metrics['clean']['images']} images) F1={metrics['clean']['macro_f1']:.4f}, "
                  f"mAP={metrics['clean']['mAP']:.4f}, Delta m={metrics['clean']['delta_m']:+.2f}", flush=True)
        selected = max(results, key=lambda m: m['delta_m'])
        (args.run / 'calibration.json').write_text(json.dumps({'trials': results, 'selected': selected}, indent=2))
        del model
        torch.cuda.empty_cache()
    subprocess.run([sys.executable, 'make_submission.py', '--run', str(args.run), '--data', str(args.data),
                    '--out', str(args.out), '--class-score-power', str(selected['class_score_power'])], check=True)
    args.out.with_suffix('.metrics.json').write_text(json.dumps({
        'checkpoint': str(args.run), 'selected_epoch': summary['val'].get('epoch'),
        'local_metrics': selected, 'official_score': None,
        'selection_note': 'Chosen using labeled training holdout only; no official validation labels used.'
    }, indent=2))


if __name__ == '__main__':
    main()
