"""Run the speckle-cand recipe on top of any trained YOLO26-P2 detector and package a submission candidate.

speckle-cand (holdout Δm +39.88) was: 100 epochs, then 40 plain epochs (learning rate 0.002), then 40 speckle epochs
(0.001, speckle on half the images), then the 1024-unit classifier refit and the cross-fitted box nudge. This script
applies the same continuation stages and post-processing to another 100-epoch detector, e.g. one started from DOTA or
COCO weights, and checks that each stage's training settings match speckle-cand's exactly.

    python models/yolo/continue_chain.py --tag dota-chain \\
        --source runs/yolo26m-p2-dota-verified-330705/train/weights/best.pt \\
        --classifier runs/yolo26m-p2-dota-verified-330705/image_classifier.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.yolo.matched_large import MEDIUM_STAGES, compare_arguments, run, write_json  # noqa: E402

STAGES = {'plain': {'epochs': 40, 'lr0': .002, 'speckle_prob': 0.},
          'speckle': {'epochs': 40, 'lr0': .001, 'speckle_prob': .5}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True, help='runs/<tag>-<stage>, results/<tag>, nonsubmissions/<date>/<tag>')
    parser.add_argument('--source', type=Path, required=True, help='trained detector to continue (best.pt)')
    parser.add_argument('--classifier', type=Path, required=True,
                        help="the source detector's image classifier (only needed by the fine-tune script's checks)")
    parser.add_argument('--stages', default='plain,speckle',
                        help='comma-separated continuation stages, in order; "plain" alone skips the speckle stage')
    args = parser.parse_args()
    if Path(args.tag).name != args.tag:
        raise ValueError('tag must be a directory name')
    stages = [name.strip() for name in args.stages.split(',') if name.strip()]
    if not stages or any(name not in STAGES for name in stages):
        raise ValueError(f'stages must come from {sorted(STAGES)}')
    for path in (args.source, args.classifier, *MEDIUM_STAGES.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    work = ROOT / 'runs' / args.tag
    work.mkdir(parents=True, exist_ok=False)
    write_json(work / 'plan.json', {'source': str(args.source), 'stages': {n: STAGES[n] for n in stages},
                                    'recipe': 'speckle-cand continuation stages and post-processing'})

    source, histories = args.source, []
    for name in stages:
        stage, tag = STAGES[name], f'{args.tag}-{name}'
        run([sys.executable, '-u', ROOT / 'models/yolo/finetune_detector.py', '--tag', tag, '--source', source,
             '--classifier', args.classifier, '--epochs', stage['epochs'], '--lr0', stage['lr0'], '--seed', 0,
             '--speckle-prob', stage['speckle_prob'], '--skip-evaluation'])
        curves = ROOT / 'results/training-curves' / tag
        compare_arguments(MEDIUM_STAGES[name], curves / 'args.yaml')  # same settings as speckle-cand's stage
        source = ROOT / 'runs' / tag / 'train/weights/best.pt'
        histories.append({'type': 'ultralytics', 'path': str((curves / 'training.csv').relative_to(ROOT)),
                          'label': f'{args.tag} {name}, {stage["epochs"]} epochs'})

    run([sys.executable, '-u', ROOT / 'models/yolo/package_candidate.py', '--tag', args.tag,
         '--detector', source, '--curves', histories[-1]['path']])
    packages = list((ROOT / 'nonsubmissions').glob(f'*/{args.tag}/{args.tag}.run.json'))
    if len(packages) != 1:
        raise RuntimeError(f'Expected one completed package, found {packages}')
    metadata = json.loads(packages[0].read_text())
    metadata['histories'] = histories  # both continuation stages, not only the last
    write_json(packages[0], metadata)
    from sarbench.twins import score
    for tag in ('speckle-cand', args.tag):
        print(f'== {tag}, one-to-many head, before the box nudge', flush=True)
        score(ROOT / 'results' / tag / 'one2many', ROOT / 'data/course-data')
    print('CHAIN_COMPLETE', packages[0].with_suffix('').with_suffix('.zip'), flush=True)


if __name__ == '__main__':
    main()
