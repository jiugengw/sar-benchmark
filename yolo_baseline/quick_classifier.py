"""Try two small image heads on frozen, officially scored YOLO features."""
from __future__ import annotations

import argparse
import csv
import copy
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'yolo_baseline'))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from train_image_head import ImageLabels, extract_features
from detector_support import load_detector
from image_head import ImageClassifierHead, classification_metrics, load_classifier_head
from sarbench.twins import clean_image_ids, score
from yolo_baseline.deadline_recovery import evaluate_export, run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-minutes', type=float, default=21)
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + args.max_minutes * 60
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('An allocated CUDA GPU is required')
    torch.set_num_threads(4)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device('cuda:0')
    weights = ROOT / 'runs/deadline-yolo26m-329644/continue_1/weights/best.pt'
    original = ROOT / 'runs/deadline-yolo26m-329644/continue_1/image_classifier.pt'
    dataset = ROOT / 'data/yolo11s-sar-512'
    split_path = ROOT / 'results/yolo26m-100-p2-328339/seed_0/split.json'
    gt_path = ROOT / 'results/yolo26m-100-p2-328339/seed_0/instances_val_holdout.json'
    split = json.loads(split_path.read_text())
    gt = json.loads(gt_path.read_text())
    clean_ids = set(clean_image_ids(gt['images']))
    train_ids = list(map(int, split['train_image_ids']))
    val_ids = list(map(int, split['val_image_ids']))
    assert len(train_ids) == 8453 and len(val_ids) == 939 and len(clean_ids) == 720
    assert not set(train_ids) & set(val_ids)
    clean_mask = torch.tensor([i in clean_ids for i in val_ids], device=device)
    names = json.loads((dataset / 'conversion_manifest.json').read_text())['classes']
    image_map = json.loads((dataset / 'image_map.json').read_text())
    with (ROOT / 'data/course-data/train/labels.csv').open() as stream:
        labels = {int(r['image_id']): int(r['label']) for r in csv.DictReader(stream)}
    def loader(ids):
        return DataLoader(ImageLabels(ids, labels, image_map, dataset), batch_size=32,
                          num_workers=4, shuffle=False, pin_memory=True)
    detector = load_detector(weights, 'yolo')
    detector.model.to(device).eval()
    detector_parameters = sum(p.numel() for p in detector.model.parameters())
    print('Extracting frozen training features (four training flip views)', flush=True)
    features, targets = extract_features(detector, loader(train_ids), device, 'yolo', 'spatial', True)
    print('Extracting single-view holdout features', flush=True)
    val_features, val_targets = extract_features(detector, loader(val_ids), device, 'yolo', 'spatial')
    del detector
    torch.cuda.empty_cache()
    extraction_seconds = time.monotonic() - started
    features, targets = features.to(device), targets.to(device)
    val_features, val_targets = val_features.to(device), val_targets.to(device)
    all_truth = val_targets.cpu().tolist()
    clean_truth = val_targets[clean_mask].cpu().tolist()
    def evaluate(head):
        head.eval()
        with torch.inference_mode():
            pred = head(val_features).argmax(1)
        return {'full': classification_metrics(all_truth, pred.cpu().tolist(), names),
                'clean': classification_metrics(clean_truth, pred[clean_mask].cpu().tolist(), names)}
    baseline_head, baseline_payload = load_classifier_head(original, device)
    baseline = evaluate(baseline_head)
    saved_baseline = json.loads((ROOT / 'results/yolo26m_p2_finetune_329644/clean_continuation_comparison.json').read_text())['runs']['continue_1']
    for subset in ['full', 'clean']:
        if abs(baseline[subset]['macro_f1'] - saved_baseline[subset]['macro_f1']) > 1e-6:
            raise RuntimeError(f'Feature-path baseline does not reproduce {subset} F1: {baseline[subset]}')
    print('BASELINE', {s: baseline[s]['macro_f1'] for s in baseline}, flush=True)
    best_clean = baseline['clean']['macro_f1']
    best_payload = None
    best_metrics = None
    history = []
    recipes = [dict(name='warm_512', hidden_dim=512, lr=0.0001, epochs=180, smoothing=0.0, warm=True),
               dict(name='wider_1024', hidden_dim=1024, lr=0.0005, epochs=240, smoothing=0.05, warm=False)]
    counts = torch.bincount(targets, minlength=len(names)).float()
    class_weights = counts.sum() / (len(names) * counts.clamp_min(1))
    for recipe in recipes:
        torch.manual_seed(0)
        head = ImageClassifierHead(features.shape[1], len(names), dropout=0.35,
                                   hidden_dim=recipe['hidden_dim']).to(device)
        if recipe['warm']:
            head.load_state_dict(baseline_payload['state_dict'])
        parameters = detector_parameters + sum(p.numel() for p in head.parameters())
        assert parameters <= 40_000_000
        optimizer = torch.optim.AdamW(head.parameters(), lr=recipe['lr'], weight_decay=0.001)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=recipe['smoothing'])
        batches = DataLoader(TensorDataset(features, targets), batch_size=256, shuffle=True,
                             generator=torch.Generator().manual_seed(0))
        for epoch in range(1, recipe['epochs'] + 1):
            # Reserve four minutes for submission-path validation and packaging.
            if time.monotonic() > deadline - 240:
                print('Training time budget reached; packaging current winner', flush=True)
                break
            head.train()
            for x, y in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(head(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite classifier loss')
                loss.backward()
                optimizer.step()
            metrics = evaluate(head)
            history.append({'recipe': recipe['name'], 'epoch': epoch,
                            'full_f1': metrics['full']['macro_f1'], 'clean_f1': metrics['clean']['macro_f1']})
            # A candidate must improve clean F1 and retain the original full F1.
            if metrics['clean']['macro_f1'] > best_clean and metrics['full']['macro_f1'] >= baseline['full']['macro_f1']:
                best_clean = metrics['clean']['macro_f1']
                best_metrics = metrics
                best_payload = copy.deepcopy(baseline_payload)
                best_payload.update(hidden_dim=recipe['hidden_dim'], dropout=0.35,
                                    state_dict={k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                                    holdout=metrics['full'])
                best_payload['training'] = {'recipe': recipe, 'best_epoch': epoch,
                    'checkpoint_selection': 'highest clean holdout macro-F1, requiring full F1 >= original',
                    'clean_holdout': metrics['clean'], 'seed': 0, 'split': str(split_path),
                    'detector_frozen': True, 'detector_checkpoint': str(weights),
                    'train_images': len(train_ids), 'training_views': len(targets),
                    'train_flips': True, 'validation_views': 1, 'total_parameters': parameters,
                    'feature_extraction_seconds': extraction_seconds}
                torch.save(best_payload, output / 'image_classifier.pt')
                print('NEW_BEST', history[-1], flush=True)
            if epoch == 1 or epoch % 20 == 0:
                print('HEAD_PROGRESS', history[-1], flush=True)
        del head, optimizer
        torch.cuda.empty_cache()
    report = {'baseline': baseline, 'selected': best_metrics, 'history': history,
              'selection': 'clean macro-F1 with no full-holdout F1 regression',
              'detector_checkpoint': str(weights), 'detector_frozen': True,
              'original_classifier': str(original), 'feature_extraction_seconds': extraction_seconds,
              'job_id': os.environ.get('SLURM_JOB_ID'), 'gpu': torch.cuda.get_device_name(0),
              'official_score': None, 'no_ensemble': True, 'no_test_time_augmentation': True}
    result_dir = ROOT / 'results' / output.name
    write_json(result_dir / 'head_comparison.json', report)
    if best_payload is None:
        print('NO_IMPROVEMENT: retain original official 34.5697 submission', flush=True)
        return
    del features, targets, val_features, val_targets, baseline_head
    torch.cuda.empty_cache()
    classifier = output / 'image_classifier.pt'
    metrics = evaluate_export(weights, classifier, 'one2many', result_dir / 'one2many')
    clean_report = score(result_dir / 'one2many', ROOT / 'data/course-data')
    write_json(result_dir / 'clean_scores.json', clean_report)
    # Reproduce the selection scores through the exact submission generator.
    for subset in ['full', 'clean']:
        assert abs(clean_report[subset]['macro_f1'] - best_metrics[subset]['macro_f1']) < 1e-6
    assert abs(metrics['mAP'] - saved_baseline['full']['mAP']) < 1e-9
    package_dir = ROOT / 'submissions' / output.name
    package_dir.mkdir(parents=True, exist_ok=True)
    package = package_dir / f'{output.name}.zip'
    run([sys.executable, '-u', ROOT / 'yolo_baseline/make_submission.py', '--weights', weights,
         '--classifier-checkpoint', classifier, '--detection-branch', 'one2many', '--out', package])
    shutil.copy2(result_dir / 'one2many/metrics.json', package_dir / 'holdout.metrics.json')
    shutil.copy2(result_dir / 'clean_scores.json', package_dir / 'clean_scores.json')
    report['zip'] = str(package)
    report['wall_seconds'] = time.monotonic() - started
    write_json(package_dir / 'candidate.json', report)
    write_json(output / 'complete.json', report)
    print('CANDIDATE_READY', package, 'clean F1 gain:', best_clean - baseline['clean']['macro_f1'], flush=True)


if __name__ == '__main__':
    main()
