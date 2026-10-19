"""Turn a trained YOLO26 detector into a submission candidate, the way the best submission was built.

1. Train the 1024-neuron spatial image classifier on the frozen detector's features (the recipe behind 96.06% F1).
2. Score the detector + classifier through the exact submission code (one-to-many head) on the full and clean holdout.
3. Cross-fit the per-class box-edge nudge on the clean holdout images (kept only if every held-out check gains).
4. Write the official-validation ZIP, with the nudge when it was kept, into nonsubmissions/<DDMMYY>/<tag>/,
   with run.json for the training curves.

    python models/yolo/package_candidate.py --tag speckle-cand --detector runs/ft40-speckle-330447/train/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(args):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True, help='name of the candidate (results/<tag>, nonsubmissions/<date>/<tag>)')
    parser.add_argument('--detector', type=Path, required=True, help='trained detector weights (best.pt or last.pt)')
    parser.add_argument('--curves', type=Path, help='per-epoch training.csv of the detector, for run.json')
    parser.add_argument('--box-audit', type=Path,
                        help='reuse this box_bias_crossfit.json instead of fitting the nudge on the holdout; '
                             'needed when the detector was trained on the holdout (all-data runs)')
    parser.add_argument('--classifier-all-data', action='store_true',
                        help='train the image classifier on all 9,392 labelled images and keep its last epoch '
                             '(for a detector trained on all data)')
    args = parser.parse_args()
    from models.yolo.deadline_recovery import evaluate_export, write_json

    detector = args.detector.resolve()
    if not detector.is_file():
        raise FileNotFoundError(detector)
    work = ROOT / 'runs' / f'{args.tag}-package'
    work.mkdir(parents=True, exist_ok=True)
    classifier = work / 'image_classifier.pt'
    run([sys.executable, '-u', ROOT / 'models/yolo/train_image_head.py',
         '--detector-checkpoint', detector, '--output', classifier,
         '--split', ROOT / 'results/yolo26m-100-p2-328339/seed_0/split.json',
         '--epochs', 240, '--batch', 32, '--head-batch', 256, '--workers', 4,
         '--feature-pooling', 'spatial', '--hidden-dim', 1024, '--dropout', 0.35,
         '--learning-rate', 0.0005, '--weight-decay', 0.001, '--train-flips',
         *(['--all-data'] if args.classifier_all_data else [])])

    holdout = ROOT / 'results' / args.tag / 'one2many'
    evaluate_export(detector, classifier, 'one2many', holdout)
    audit = ROOT / 'results' / args.tag / 'box_bias_crossfit.json'
    if args.box_audit:  # offsets fitted on a model that never saw the holdout
        audit.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.box_audit, audit)
    else:
        run([sys.executable, '-u', ROOT / 'models/yolo/quick_box_bias.py', '--results', holdout, '--out', audit])
    nudge = json.loads(audit.read_text()).get('accepted', False)

    base_zip = work / f'{args.tag}_base.zip'
    run([sys.executable, '-u', ROOT / 'models/yolo/make_submission.py', '--weights', detector,
         '--classifier-checkpoint', classifier, '--detection-branch', 'one2many', '--out', base_zip])
    package_dir = ROOT / 'nonsubmissions' / time.strftime('%d%m%y') / args.tag
    if nudge:  # package_box_bias.py writes nonsubmissions/<date>/<tag>/<tag>.zip and results/<tag>/seed_0/
        run([sys.executable, '-u', ROOT / 'models/yolo/package_box_bias.py', '--tag', args.tag,
             '--base-zip', base_zip, '--results', holdout, '--audit', audit])
    else:
        package_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_zip, package_dir / f'{args.tag}.zip')
    shutil.copy2(classifier, package_dir / 'image_classifier.pt')
    shutil.copy2(holdout / 'metrics.json', package_dir / 'holdout.metrics.json')
    shutil.copy2(audit, package_dir / 'box_bias_crossfit.json')
    histories = []
    if args.curves and (ROOT / args.curves).is_file():
        histories.append({'type': 'ultralytics', 'path': str(args.curves), 'label': f'{args.tag} detector'})
    write_json(package_dir / f'{args.tag}.run.json', {
        'zip': f'{args.tag}.zip', 'histories': histories,
        'note': f'1024-neuron classifier refit; box nudge {"applied" if nudge else "rejected by cross-fit"}',
        **({} if histories else {'missing': 'pass --curves to record the detector training history'})})
    write_json(package_dir / 'candidate.json', {
        'detector': str(detector), 'classifier': str(classifier), 'branch': 'one2many', 'box_nudge': nudge,
        'no_ensemble': True, 'no_test_time_augmentation': True, 'image_size': 512, 'official_score': None})
    from sarbench.twins import score
    print(f'== {args.tag}: detector + refit classifier, before the box nudge', flush=True)
    score(holdout, ROOT / 'data/course-data')
    print(f'CANDIDATE {package_dir / (args.tag + ".zip")}  (box nudge {"applied" if nudge else "not applied"})', flush=True)


if __name__ == '__main__':
    main()
