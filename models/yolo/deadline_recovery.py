"""Continue E5 detection training and fit a separate spatial image head before the deadline.

Each candidate uses one shared detector backbone, one image head, one 512px input,
and one selected detection branch. The labeled holdout never enters training.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(command):
    print('COMMAND', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def evaluate_export(weights, classifier, branch, output, batch=16):
    """Evaluate exactly the same prediction code that produces the official ZIP."""
    from sarbench.metrics import detection_metrics
    from models.yolo.image_head import classification_metrics

    ground_truth_path = ROOT / 'results/yolo26m-100-p2-328339/seed_0/instances_val_holdout.json'
    ground_truth = json.loads(ground_truth_path.read_text())
    with (ROOT / 'data/course-data/train/labels.csv').open(newline='') as stream:
        truth = {int(row['image_id']): int(row['label']) for row in csv.DictReader(stream)}
    class_names = [row['name'] for row in sorted(ground_truth['categories'], key=lambda row: row['id'])]
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sar-deadline-holdout-') as temporary:
        view = Path(temporary) / 'data'
        (view / 'train').mkdir(parents=True)
        (view / 'val').mkdir()
        (view / 'train/labels.csv').symlink_to(ROOT / 'data/course-data/train/labels.csv')
        for image in ground_truth['images']:
            target = view / 'val' / image['file_name']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(ROOT / 'data/course-data/train' / image['file_name'])
        write_json(view / 'val/images.json', {'images': ground_truth['images']})
        package = Path(temporary) / 'holdout.zip'
        run([sys.executable, '-u', ROOT / 'models/yolo/make_submission.py',
             '--weights', weights, '--classifier-checkpoint', classifier,
             '--detection-branch', branch, '--data', view, '--out', package, '--batch', batch])
        with zipfile.ZipFile(package) as archive:
            predictions = json.loads(archive.read('detection.json'))
            labels = {int(row['image_id']): int(row['label']) for row in csv.DictReader(
                io.StringIO(archive.read('classification.csv').decode()))}
        inference = json.loads(package.with_suffix('.inference.json').read_text())
    image_ids = [int(image['id']) for image in ground_truth['images']]
    if set(labels) != set(image_ids):
        raise RuntimeError('Submission-path validation did not cover the saved holdout exactly')
    classifier_metrics = classification_metrics([truth[i] for i in image_ids], [labels[i] for i in image_ids], class_names)
    metrics = detection_metrics(str(ground_truth_path), predictions, image_ids)
    metrics.update({key: classifier_metrics[key] for key in ('accuracy', 'macro_f1', 'balanced_accuracy')})
    metrics['image_classification'] = classifier_metrics
    metrics['combined_delta_m'] = 50 * (metrics['macro_f1'] / 0.9331 + metrics['mAP'] / 0.2613 - 2)
    write_json(output / 'metrics.json', {'val': metrics, 'submission_pipeline': inference,
                                       'split': '939-image labeled training holdout; no official labels'})
    write_json(output / 'predictions_val.json', predictions)
    write_json(output / 'image_classification_predictions.json', labels)
    shutil.copy2(ground_truth_path, output / 'instances_val_holdout.json')
    print('EXACT_HOLDOUT_RESULT', branch, {key: metrics[key] for key in ('macro_f1', 'mAP', 'combined_delta_m')}, flush=True)
    return metrics


def candidate(weights, stage, output, tag, feature_batch=32):
    classifier = output / stage / 'image_classifier.pt'
    classifier.parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, '-u', ROOT / 'models/yolo/train_image_head.py',
         '--detector-checkpoint', weights, '--output', classifier,
         '--split', ROOT / 'results/yolo26m-100-p2-328339/seed_0/split.json',
         '--epochs', 100, '--batch', feature_batch, '--head-batch', 256, '--workers', 4,
         '--feature-pooling', 'spatial', '--hidden-dim', 512, '--dropout', 0.35,
         '--learning-rate', 0.0005, '--weight-decay', 0.001, '--train-flips'])
    branches = {}
    results_root = ROOT / 'results' / tag / stage
    for branch in ('one2one', 'one2many'):
        branches[branch] = evaluate_export(weights, classifier, branch, results_root / branch)
    selected = max(branches, key=lambda key: branches[key]['combined_delta_m'])
    submission_root = ROOT / 'nonsubmissions' / tag / stage
    submission_root.mkdir(parents=True, exist_ok=True)
    package = submission_root / f'{tag}_{stage}.zip'
    run([sys.executable, '-u', ROOT / 'models/yolo/make_submission.py',
         '--weights', weights, '--classifier-checkpoint', classifier,
         '--detection-branch', selected, '--out', package])
    shutil.copy2(classifier.with_suffix('.metrics.json'), submission_root / 'classifier.metrics.json')
    shutil.copy2(results_root / selected / 'metrics.json', submission_root / 'holdout.metrics.json')
    write_json(submission_root / 'candidate.json', {
        'weights': str(weights), 'classifier': str(classifier), 'branch': selected,
        'holdout': {key: branches[selected][key] for key in ('macro_f1', 'mAP', 'combined_delta_m')},
        'official_score': None, 'no_ensemble': True, 'no_test_time_augmentation': True,
        'image_size': 512, 'training_split': '8453 images; fixed 939-image holdout excluded',
    })
    (submission_root / 'scoring_result').mkdir(exist_ok=True)
    (submission_root / 'scoring_result/.gitkeep').touch()
    result = {'stage': stage, 'weights': str(weights), 'classifier': str(classifier),
              'branch': selected, 'zip': str(package), 'holdout_delta_m': branches[selected]['combined_delta_m']}
    write_json(output / f'{stage}_ready.json', result)
    print('CANDIDATE_READY', json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=40, help='epochs in each of two continuation stages')
    parser.add_argument('--stages', type=int, default=2)
    parser.add_argument('--skip-baseline-head', action='store_true')
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError('An allocated CUDA GPU is required')
    torch.set_num_threads(4)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    tag = args.output.name
    weights = ROOT / 'models/yolo/phase3_assets/checkpoints/E5_yolo26m_p2_best.pt'
    write_json(args.output / 'plan.json', {
        'source': str(weights), 'continuation_epochs_per_stage': args.epochs,
        'stages': args.stages, 'seed': 0, 'gpu': torch.cuda.get_device_name(0),
        'strategy': 'detector-only continuation; spatial image head fitted on frozen detector features',
        'initialization': 'existing course-trained E5; original external initialization COCO, no external SAR model',
        'benchmark_limits': {'image_size': 512, 'max_total': 130000000, 'max_trainable': 40000000,
                             'ensemble': False, 'test_time_augmentation': False},
    })
    candidates = []
    if not args.skip_baseline_head:
        candidates.append(candidate(weights, 'e5_spatial_head', args.output, tag))
    for index in range(1, args.stages + 1):
        stage = f'continue_{index}'
        detector = YOLO(str(weights))
        total = sum(parameter.numel() for parameter in detector.model.parameters())
        if total > 40_000_000:
            raise RuntimeError(f'Detector exceeds full-training budget: {total}')
        started = time.monotonic()
        detector.train(
            data=str(ROOT / 'data/yolo11s-sar-512/dataset.yaml'),
            epochs=args.epochs, imgsz=512, batch=16, workers=4, device=0,
            pretrained=True, optimizer='MuSGD', lr0=0.002 if index == 1 else 0.001,
            lrf=0.1, momentum=0.9, weight_decay=0.0005, warmup_epochs=1,
            warmup_bias_lr=0.0, cos_lr=True, seed=0, deterministic=True,
            mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0,
            translate=0.05, scale=0.2, fliplr=0.5, flipud=0.5,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
            multi_scale=0.0, amp=True, patience=args.epochs + 1,
            project=str(args.output), name=stage, plots=True, save_period=10,
        )
        stage_root = args.output / stage
        weights = stage_root / 'weights/best.pt'
        write_json(stage_root / 'compute.json', {'wall_seconds': time.monotonic() - started,
                                                'detector_parameters': total, 'epochs': args.epochs})
        del detector
        torch.cuda.empty_cache()
        candidates.append(candidate(weights, stage, args.output, tag))
        write_json(args.output / 'candidates.json', candidates)
    write_json(args.output / 'complete.json', {'candidates': candidates,
               'best_local': max(candidates, key=lambda row: row['holdout_delta_m'])})
    print('DEADLINE_RECOVERY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
