"""Does averaging the weights of a run's last checkpoints beat its best single checkpoint? Holdout only, no training.

Weight averaging (SWA, Zhang et al. 2020, "SWA Object Detection", arXiv 2012.12645: about +1 AP on COCO for six
detectors) merges several checkpoints of one run into ONE model of the same size before submission, so prediction
is still one model and one pass per image. Checkpoints late in a run sit around the same minimum; their average
tends to lie nearer its centre, which generalises better than any single one. Their recipe trains 12 extra epochs
with a cyclic learning rate; this first look only averages checkpoints the run already saved (save_period 10).

    python models/yolo/swa_study.py --run runs/ft40-memorise-lr2-332326 \\
        --classifier runs/memorise-lr2-332326-package/image_classifier.pt --tag swa-memorise-lr2
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))

COMBINATIONS = ('best', 'last', 'last+epoch30', 'last+epoch30+epoch20', 'last+epoch30+epoch20+epoch10', 'last+best')


def checkpoint_model(path: Path):
    import torch

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    return checkpoint, (checkpoint.get('ema') or checkpoint['model']).float()


def average(paths: list[Path], output: Path) -> Path:
    """Average the weights (and batch-norm statistics) of these checkpoints into one checkpoint like the first."""
    import torch

    template, model = checkpoint_model(paths[0])
    states = [model.state_dict()] + [checkpoint_model(p)[1].state_dict() for p in paths[1:]]
    merged = {}
    for key, value in states[0].items():
        if value.is_floating_point():
            merged[key] = sum(s[key] for s in states) / len(states)
        else:
            merged[key] = value  # counters such as num_batches_tracked
    model.load_state_dict(merged)
    checkpoint = copy.copy(template)
    checkpoint.update(model=copy.deepcopy(model).half(), ema=None, optimizer=None,
                      averaged_from=[str(p) for p in paths])
    torch.save(checkpoint, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True, help='Ultralytics fine-tune folder (has train/weights)')
    parser.add_argument('--classifier', type=Path, required=True, help='image classifier used for scoring only')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--combinations', help='comma-separated checkpoint sets, names joined by +, e.g. '
                                               '"best,last+epoch10"; "epochs:A-B" averages epochA..epochB plus last')
    args = parser.parse_args()

    from models.yolo.deadline_recovery import evaluate_export
    from sarbench.twins import score

    weights = args.run / 'train/weights'
    out = ROOT / 'results' / args.tag
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    combinations = COMBINATIONS
    if args.combinations:
        combinations = []
        for item in args.combinations.split(','):
            if item.startswith('epochs:'):
                first, last = map(int, item[len('epochs:'):].split('-'))
                item = '+'.join(['last'] + [f'epoch{e}' for e in range(first, last + 1)])
            combinations.append(item)
    for combination in combinations:
        paths = [weights / f'{name}.pt' for name in combination.split('+')]
        if not all(p.is_file() for p in paths):
            print(f'skip {combination}: missing checkpoint', flush=True)
            continue
        name = combination if len(combination) < 60 else f'average-of-{len(paths)}'
        model = paths[0] if len(paths) == 1 else average(paths, out / f'{combination}.pt')
        evaluate_export(model, args.classifier, 'one2many', out / name)
        print(f'== {combination}', flush=True)
        report = score(out / name, ROOT / 'data/course-data')
        rows.append({'checkpoints': combination, 'full_mAP': report['full']['mAP'], 'clean_mAP': report['clean']['mAP'],
                     'clean_AP75': report['clean']['AP75'], 'clean_per_class_AP': report['clean']['per_class_AP']})
        if len(paths) > 1:
            (out / f'{combination}.pt').rename(out / f'average-{len(rows)}.pt')  # kept: a winner gets packaged  # 40 MB each; rebuild from the run's checkpoints if a winner is kept
        (out / 'swa_study.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f"\n{'checkpoints':30s} {'full mAP':>8s} {'clean mAP':>9s} {'clean AP75':>10s}")
    for r in rows:
        print(f"{r['checkpoints'][:30]:30s} {100 * r['full_mAP']:8.2f} {100 * r['clean_mAP']:9.2f} {100 * r['clean_AP75']:10.2f}")


if __name__ == '__main__':
    main()
