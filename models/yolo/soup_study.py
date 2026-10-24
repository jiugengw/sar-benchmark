"""Model soup: average the weights of several fine-tunes that start from the SAME checkpoint into ONE model.

Wortsman et al. 2022 ("Model soups", arXiv 2203.05482): fine-tunes of one pretrained model with different
hyperparameters often lie in one low-loss basin, and their weight average beats each of them, at the cost of one
model and one pass per image. Our memorise runs all start from the x4 third stage and differ only in beta, speckle
or extra views, so they are soup candidates. Each soup is evaluated on the holdout exactly like a submission
(deadline_recovery.evaluate_export).

    python models/yolo/soup_study.py --tag soup-bp4 --soup beta8+beta12=runs/a/train/weights/best.pt,runs/b/...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/yolo'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--classifier', type=Path, default=ROOT / 'runs/third-stage-331789-package/image_classifier.pt')
    parser.add_argument('--soup', action='append', required=True, help='name=weights1,weights2,...')
    args = parser.parse_args()
    from models.yolo.deadline_recovery import evaluate_export
    from swa_study import average

    out = ROOT / 'runs' / args.tag
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.soup:
        name, paths = spec.split('=', 1)
        paths = [ROOT / p for p in paths.split(',')]
        model = average(paths, out / f'{name}.pt')
        evaluate_export(model, args.classifier, 'one2many', ROOT / 'results' / f'{args.tag}-{name}' / 'one2many')
        print(f'SOUP_READY {name}: {len(paths)} ingredients', flush=True)


if __name__ == '__main__':
    main()
