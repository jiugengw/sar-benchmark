"""Prepare or run five independent offline training folds, then summarize their scores."""
import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from sarbench.cross_validation import stratified_folds, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/course-data'))
    parser.add_argument('--out', type=Path, default=Path('runs/anchor48-cv5'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--run', action='store_true', help='execute training sequentially on one GPU')
    parser.add_argument('--summarize-only', action='store_true')
    args, training_args = parser.parse_known_args()
    if args.run and args.summarize_only:
        parser.error('--run and --summarize-only are mutually exclusive')
    # Partition and output options are owned by this runner to keep folds disjoint.
    reserved = {'--out', '--results', '--checkpoints', '--fold', '--folds', '--split-seed', '--limit'}
    if any(arg.split('=')[0] in reserved for arg in training_args):
        parser.error('Do not override fold/output options or --limit in training arguments')
    with (args.data / 'train/labels.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    labels = [int(row['label']) for row in rows]
    partitions = stratified_folds(labels, args.folds, args.split_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    defaults = ['--backbone', 'terramind', '--init', 'pretrained', '--adapt', 'moelora',
                '--epochs', '48', '--batch-size', '16', '--lr', '1e-4', '--seed', '0',
                '--workers', '4', '--eval-every', '2', '--box-loss-weight', '2',
                '--photometric', '--anchor-profile', 'default']
    commands, fold_reports = [], []
    for fold, (train, val) in enumerate(partitions):
        command = [sys.executable, '-u', str(Path(__file__).with_name('train.py')),
                   *defaults, *training_args, '--data', str(args.data), '--fold', str(fold),
                   '--folds', str(args.folds), '--split-seed', str(args.split_seed),
                   '--out', str(args.out), '--results', str(args.out / 'results'),
                   '--checkpoints', str(args.out / 'checkpoints')]
        commands.append(command)
        fold_reports.append({'fold': fold, 'train_count': len(train), 'val_count': len(val),
                             'val_class_counts': dict(Counter(labels[i] for i in val)),
                             'train_indices': train, 'val_indices': val})
        print(f'Fold {fold}: {len(train)} train / {len(val)} validation', flush=True)
    plan = {'data': str(args.data.resolve()), 'split_seed': args.split_seed,
            'folds': fold_reports, 'commands': commands}
    plan_path = args.out / 'fold_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        parser.error('Existing output has a different plan; choose a new --out directory')
    plan_path.write_text(json.dumps(plan, indent=2))
    if args.run:
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA GPU unavailable; folds are prepared but training cannot start')
        for fold, command in enumerate(commands):
            existing = list((args.out / f'fold_{fold}').glob('*/metrics.json'))
            if existing:
                print(f'Fold {fold} already completed; skipping', flush=True)
                continue
            with (args.out / f'fold_{fold}.console.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.run or args.summarize_only:
        scores = []
        for fold in range(args.folds):
            files = list((args.out / f'fold_{fold}').glob('*/metrics.json'))
            if len(files) != 1:
                parser.error(f'Fold {fold} needs exactly one completed metrics.json; found {len(files)}')
            result = json.loads(files[0].read_text())
            scores.append(result['val'])
        summary = {'fold_scores': scores, 'aggregate': summarize(scores),
                   'std_definition': 'sample standard deviation across folds'}
        (args.out / 'cv_summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary['aggregate'], indent=2))
    else:
        print(f'Prepared {plan_path}. Add --run to train all folds sequentially.')


if __name__ == '__main__':
    main()
