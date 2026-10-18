"""Run the speckle-cand recipe on top of any trained YOLO26-P2 detector and package a submission candidate.

speckle-cand (holdout Δm +39.88) was: 100 epochs, then 40 plain epochs (learning rate 0.002), then 40 speckle epochs
(0.001, speckle on half the images), then the 1024-unit classifier refit and the cross-fitted box nudge. This script
applies the same continuation stages and post-processing to another 100-epoch detector, e.g. one started from DOTA or
COCO weights, and checks that each stage's training settings match speckle-cand's exactly.

--all-data trains both stages on the training split and the 939 holdout images together (the final-model recipe) and
keeps the last epoch. Its holdout scores are then optimistic, so the box nudge is not refitted: pass --box-audit with
the box_bias_crossfit.json of the same chain run without --all-data, and judge the result by its official score.

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
from models.yolo.matched_large import MEDIUM_STAGES, PATH_ARGUMENTS, compare_arguments, run, write_json  # noqa: E402

STAGES = {'plain': {'epochs': 40, 'lr0': .002, 'speckle_prob': 0.},
          'speckle': {'epochs': 40, 'lr0': .001, 'speckle_prob': .5}}


def same_settings(reference: Path, candidate: Path, ignore: set[str]) -> None:
    import yaml

    a, b = [yaml.safe_load(p.read_text()) for p in (reference, candidate)]
    skip = PATH_ARGUMENTS | ignore
    differences = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys() if k not in skip and a.get(k) != b.get(k)}
    if differences:
        raise ValueError(f'Training settings differ: {reference} vs {candidate}: {differences}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True, help='runs/<tag>-<stage>, results/<tag>, nonsubmissions/<date>/<tag>')
    parser.add_argument('--source', type=Path, required=True, help='trained detector to continue (best.pt)')
    parser.add_argument('--classifier', type=Path, required=True,
                        help="the source detector's image classifier (only needed by the fine-tune script's checks)")
    parser.add_argument('--stages', default='plain,speckle',
                        help='comma-separated continuation stages, in order; "plain" alone skips the speckle stage')
    parser.add_argument('--all-data', action='store_true', help='also train on the holdout images (see above)')
    parser.add_argument('--box-audit', type=Path, help='box nudge to reuse (required with --all-data)')
    parser.add_argument('--data', type=Path, help='dataset YAML for both stages (default: the standard 512 px dataset)')
    args = parser.parse_args()
    if args.all_data and not (args.box_audit and args.box_audit.is_file()):
        raise ValueError('--all-data needs --box-audit: the box_bias_crossfit.json of the matching held-out chain')
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
                                    'all_data': args.all_data, 'box_audit': str(args.box_audit),
                                    'recipe': 'speckle-cand continuation stages and post-processing'})

    source, histories = args.source, []
    for name in stages:
        stage, tag = STAGES[name], f'{args.tag}-{name}'
        run([sys.executable, '-u', ROOT / 'models/yolo/finetune_detector.py', '--tag', tag, '--source', source,
             '--classifier', args.classifier, '--epochs', stage['epochs'], '--lr0', stage['lr0'], '--seed', 0,
             '--speckle-prob', stage['speckle_prob'], '--skip-evaluation', *(['--all-data'] if args.all_data else []),
             *(['--data', args.data] if args.data else [])])
        curves = ROOT / 'results/training-curves' / tag
        if args.all_data or args.data:  # identical settings apart from the training images
            same_settings(MEDIUM_STAGES[name], curves / 'args.yaml', ignore={'data'})
        else:
            compare_arguments(MEDIUM_STAGES[name], curves / 'args.yaml')  # same settings as speckle-cand's stage
        source = ROOT / 'runs' / tag / ('train/weights/last.pt' if args.all_data else 'train/weights/best.pt')
        histories.append({'type': 'ultralytics', 'path': str((curves / 'training.csv').relative_to(ROOT)),
                          'label': f'{args.tag} {name}, {stage["epochs"]} epochs'})

    run([sys.executable, '-u', ROOT / 'models/yolo/package_candidate.py', '--tag', args.tag,
         '--detector', source, '--curves', histories[-1]['path'],
         *(['--box-audit', args.box_audit] if args.box_audit else [])])
    packages = list((ROOT / 'nonsubmissions').glob(f'*/{args.tag}/{args.tag}.run.json'))
    if len(packages) != 1:
        raise RuntimeError(f'Expected one completed package, found {packages}')
    metadata = json.loads(packages[0].read_text())
    metadata['histories'] = histories  # both continuation stages, not only the last
    write_json(packages[0], metadata)
    from sarbench.twins import score
    for tag in ('speckle-cand', args.tag):
        note = ' (OPTIMISTIC: trained on these holdout images)' if args.all_data and tag == args.tag else ''
        print(f'== {tag}, one-to-many head, before the box nudge{note}', flush=True)
        score(ROOT / 'results' / tag / 'one2many', ROOT / 'data/course-data')
    print('CHAIN_COMPLETE', packages[0].with_suffix('').with_suffix('.zip'), flush=True)


if __name__ == '__main__':
    main()
