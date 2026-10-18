"""Apply the best local medium recipe to large, checking that architecture scale is the only recipe change.

Reuse the matched 100-epoch COCO baselines, then run 40 plain epochs, 40 speckle
epochs, the same 1024-unit classifier refit, and the same cross-fitted box correction.
The medium reference is the completed speckle-cand pipeline, not an old classifier.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASELINES = {'m': 'yolo26m-100-p2-328339', 'l': 'yolo26l-100-p2-329019'}
MEDIUM_STAGES = {'plain': ROOT / 'runs/deadline-yolo26m-329644/continue_1/args.yaml',
                 'speckle': ROOT / 'results/training-curves/ft40-speckle-330447/args.yaml'}
PATH_ARGUMENTS = {'model', 'project', 'save_dir', 'name'}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def compare_arguments(reference: Path, candidate: Path) -> None:
    a, b = [yaml.safe_load(p.read_text()) for p in (reference, candidate)]
    differences = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys()
                   if k not in PATH_ARGUMENTS and a.get(k) != b.get(k)}
    if differences:
        raise ValueError(f'Training settings differ: {reference} vs {candidate}: {differences}')


def plan(tag: str) -> dict:
    if Path(tag).name != tag or tag in {'.', '..'}:
        raise ValueError('tag must be a directory name')
    curves = ROOT / 'results/training-curves'
    compare_arguments(curves / BASELINES['m'] / 'args.yaml', curves / BASELINES['l'] / 'args.yaml')
    splits = [json.loads((ROOT / 'results' / name / 'seed_0/split.json').read_text())
              for name in BASELINES.values()]
    if splits[0] != splits[1]:
        raise ValueError('Baseline train/holdout splits differ')
    for scale, name in BASELINES.items():
        with (curves / name / 'training.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        best = max(rows, key=lambda r: float(r['metrics/mAP50-95(B)']))
        if len(rows) != 100 or int(best['epoch']) != 100:
            raise ValueError(f'{scale} baseline is not the expected 100-epoch best checkpoint')
    source = ROOT / 'runs' / BASELINES['l'] / 'train/weights/best.pt'
    classifier = ROOT / 'runs' / BASELINES['l'] / 'image_classifier.pt'
    for path in (source, classifier, *MEDIUM_STAGES.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    medium_experiment = json.loads((curves / 'ft40-speckle-330447/experiment.json').read_text())
    if any(medium_experiment[k] != v for k, v in
           {'epochs': 40, 'lr0': .001, 'nwd_weight': 0., 'speckle_prob': .5, 'speckle_looks': 4.}.items()):
        raise ValueError('Medium speckle reference does not match the planned recipe')
    return {'tag': tag, 'source': str(source), 'classifier_for_intermediate_stages': str(classifier),
            'baseline_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'baseline_argument_check': 'identical except model/project/save_dir/name',
            'split_check': 'identical 8453 train / 939 holdout',
            'medium_reference': 'speckle-cand', 'architecture_change': 'YOLO26m-P2 to YOLO26l-P2',
            'seed': 0, 'baseline_epochs': 100, 'total_detector_epochs': 180,
            'stages': [{'name': 'plain', 'epochs': 40, 'lr0': .002, 'speckle_prob': 0.},
                       {'name': 'speckle', 'epochs': 40, 'lr0': .001, 'speckle_prob': .5, 'speckle_looks': 4.}],
            'postprocessing': 'package_candidate.py: 1024-unit classifier, 240 epochs, then twin-grouped box cross-fit',
            'primary_comparison': 'one2many exported detection mAP, full and clean, before calibration',
            'secondary_comparison': 'refitted classifier F1 and delta_m; calibrated scores include fit images',
            'scope': 'single seed; size-specific COCO initializations; no causal claim across seeds'}


def run(command: list) -> None:
    print('COMMAND', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    recipe = plan(args.tag)
    print(json.dumps(recipe, indent=2), flush=True)
    if args.check_only:
        return
    work = ROOT / 'runs' / args.tag
    work.mkdir(exist_ok=False)
    write_json(work / 'plan.json', recipe)
    source = Path(recipe['source'])
    histories = []
    for stage in recipe['stages']:
        name = f'{args.tag}-{stage["name"]}'
        if (ROOT / 'runs' / name).exists():
            raise FileExistsError(f'Will not overwrite runs/{name}')
        command = [sys.executable, '-u', ROOT / 'models/yolo/finetune_detector.py',
                   '--tag', name, '--source', source,
                   '--classifier', recipe['classifier_for_intermediate_stages'],
                   '--epochs', stage['epochs'], '--lr0', stage['lr0'], '--seed', recipe['seed'],
                   '--speckle-prob', stage['speckle_prob'], '--skip-evaluation']
        run(command)
        history = ROOT / 'results/training-curves' / name
        compare_arguments(MEDIUM_STAGES[stage['name']], history / 'args.yaml')
        source = ROOT / 'runs' / name / 'train/weights/best.pt'
        histories.append({'type': 'ultralytics', 'path': str((history / 'training.csv').relative_to(ROOT)),
                          'label': f'YOLO26l-P2 {stage["name"]}, 40 epochs'})
        write_json(work / f'{stage["name"]}_complete.json', {'weights': str(source), 'settings_match_medium': True})
    run([sys.executable, '-u', ROOT / 'models/yolo/package_candidate.py', '--tag', args.tag,
         '--detector', source, '--curves', histories[-1]['path']])
    # Include both continuation stages in the packaged training history.
    packages = list((ROOT / 'nonsubmissions').glob(f'*/{args.tag}/{args.tag}.run.json'))
    if len(packages) != 1:
        raise RuntimeError(f'Expected one completed package, found {packages}')
    metadata = json.loads(packages[0].read_text())
    metadata['histories'] = histories
    metadata['comparison_plan'] = recipe
    write_json(packages[0], metadata)
    from sarbench.twins import score
    comparison = {'plan': recipe, 'before_box_calibration': {}}
    for scale, tag in [('medium', 'speckle-cand'), ('large', args.tag)]:
        print(f'== matched comparison: {scale}, before calibration', flush=True)
        comparison['before_box_calibration'][scale] = score(ROOT / 'results' / tag / 'one2many',
                                                          ROOT / 'data/course-data')
    write_json(ROOT / 'results' / args.tag / 'architecture_comparison.json', comparison)
    write_json(work / 'complete.json', {'package': str(packages[0].with_suffix('').with_suffix('.zip')),
                                         'matched_recipe': True, 'plan': recipe})
    print('MATCHED_LARGE_COMPLETE', args.tag, flush=True)


if __name__ == '__main__':
    main()
