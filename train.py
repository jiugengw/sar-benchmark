"""Train classification and detection jointly, then score a holdout of labeled training images.
Example: python train.py --data data/course-data --backbone vit --init pretrained --adapt lora"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from sarbench.adapters import add_adapters, unfreeze_last_blocks
from sarbench.backbones import build_backbone
from sarbench.data import SARMultiTask, collate, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from sarbench.model import MultiTaskModel
from sarbench.optim import parameter_groups
from sarbench.inference import calibrate_detections

# Validation reference: mean ViT + LoRA scores over seeds 0, 1 and 2.
VAL_REFERENCE_F1 = 0.9331
VAL_REFERENCE_MAP = 0.2613


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset' / 'SARFact-Course-20K')
    p.add_argument('--backbone', required=True, choices=['vit', 'terramind'])
    p.add_argument('--init', required=True, choices=['pretrained', 'scratch'])
    p.add_argument('--backbone-weights', type=Path,
                   help='local pretrained ViT backbone state dict; avoids a Hub download')
    p.add_argument('--adapt', required=True, choices=['full', 'lora', 'moelora'])
    p.add_argument('--unfreeze-last', type=int, default=0,
                   help='also train the last N encoder blocks with pretrained LoRA/MoE-LoRA (default: 0)')
    p.add_argument('--unfrozen-lr', type=float, default=2e-5,
                   help='learning rate for original weights enabled by --unfreeze-last')
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone-lr', type=float, help='for the trainable backbone parameters (default: --lr)')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--eval-every', type=int, default=4, help='validate every N epochs, and after the last one')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='use the first N labeled training images before splitting (quick checks)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fold', type=int, help='zero-based stratified validation fold (enables cross-validation)')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--split-seed', type=int, default=0, help='fixed cross-validation partition seed')
    p.add_argument('--photometric', action='store_true', help='training-only mild brightness/contrast jitter')
    p.add_argument('--select-best', action='store_true', help='save the evaluated epoch with highest local Delta m')
    p.add_argument('--anchor-profile', choices=['default', 'wide', 'large'], default='default')
    p.add_argument('--geometry', choices=['stretch', 'pad'], default='stretch')
    p.add_argument('--class-sampling', choices=['uniform', 'sqrt-balanced'], default='uniform')
    p.add_argument('--box-loss-weight', type=float, default=1.0,
                   help='multiplier for RoI box regression loss; classification and RPN losses unchanged')
    p.add_argument('--out', type=Path, default=Path('runs'))
    p.add_argument('--results', type=Path, default=Path('results'),
                   help='parent folder for saved JSON results, organized by configuration and seed')
    p.add_argument('--checkpoints', type=Path, default=Path('checkpoints'),
                   help='parent folder for preserved model weights and configuration, organized by seed')
    args = p.parse_args()
    if args.folds < 2 or (args.fold is not None and not 0 <= args.fold < args.folds):
        p.error('--folds must be >= 2 and --fold must be in [0, --folds)')
    if not math.isfinite(args.box_loss_weight) or args.box_loss_weight <= 0:
        p.error('--box-loss-weight must be finite and positive')
    if args.init == 'scratch' and args.adapt != 'full':
        p.error('--init scratch is only valid with --adapt full')
    if args.unfreeze_last < 0 or args.unfreeze_last > 12:
        p.error('--unfreeze-last must be between 0 and 12')
    if args.unfreeze_last and (args.init != 'pretrained' or args.adapt == 'full'):
        p.error('--unfreeze-last requires pretrained initialization and lora or moelora adaptation')
    if args.backbone_lr is None:
        args.backbone_lr = args.lr
    return args


def log(message, file):
    """Print to the console and to the run's log.txt."""
    print(message, flush=True)
    print(message, file=file, flush=True)


def make_loader(args, indices, train):
    """Build a loader over selected rows from the labeled course training set."""
    dataset = SARMultiTask(args.data, train=train, limit=args.limit, indices=indices, geometry=args.geometry,
                           photometric=getattr(args, 'photometric', False))
    sampler = None
    if train and args.class_sampling == 'sqrt-balanced':
        labels = torch.tensor([int(row['label']) for row in dataset.rows])
        counts = torch.bincount(labels, minlength=9).clamp_min(1)
        weights = counts[labels].double().rsqrt()
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True,
                                         generator=torch.Generator().manual_seed(args.seed))
    return DataLoader(dataset, args.batch_size, shuffle=train and sampler is None, sampler=sampler,
                      num_workers=args.workers, collate_fn=collate,
                      persistent_workers=train and args.workers > 0)

def lr_factor(step, warmup, total):
    """Learning-rate multiplier: linear warmup over `warmup` steps, then cosine decay to 0 at step `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))  # --epochs 1: no decay phase


def train_one_epoch(model, loader, optimizer, schedule, epoch, log_file, box_loss_weight=1.0):
    """One pass over the training set; the loss is cross-entropy plus the four Faster R-CNN losses."""
    model.train()
    start, seen = time.time(), 0
    for step, (images, labels, targets) in enumerate(loader, 1):
        targets = [{'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda()} for t in targets]
        with torch.autocast('cuda', dtype=torch.bfloat16):  # bf16 has fp32's range: no loss scaling needed
            logits, _, det_losses = model(images.cuda(), targets)
            losses = {'classification': F.cross_entropy(logits, labels.cuda()), **det_losses}
        loss = sum(value * (box_loss_weight if name == 'loss_box_reg' else 1.0)
                   for name, value in losses.items())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        seen += len(images)
        if step % 50 == 0 or step == len(loader):
            parts = ' '.join(f'{name} {value.item():.3f}' for name, value in losses.items())
            log(f'epoch {epoch} iter {step}/{len(loader)} loss {loss.item():.3f} ({parts}) '
                f'lr {optimizer.param_groups[1]["lr"]:.2e} {seen / (time.time() - start):.1f} images/s', log_file)
            start, seen = time.time(), 0


@torch.no_grad()
def evaluate(model, loader, gt_json, class_score_power=0.0):
    """Classification and COCO detection metrics on one split, and the detections in COCO result format."""
    model.eval()
    labels, predicted, detections, image_ids = [], [], [], []
    for images, y, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.cuda())
        outputs = calibrate_detections(outputs, logits, class_score_power)
        labels += y.tolist()
        predicted += logits.argmax(dim=1).tolist()
        for target, out in zip(targets, outputs):
            image_ids.append(target['image_id'])
            boxes = to_original_xywh(out['boxes'], target['orig_size'], geometry=target['geometry'])
            valid = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
            detections += [{'image_id': target['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                           for b, c, s in zip(boxes[valid].tolist(), out['labels'][valid].tolist(),
                                             out['scores'][valid].tolist())]
    metrics = {**classification_metrics(labels, predicted), **detection_metrics(gt_json, detections, image_ids)}
    return metrics, detections


def headline(metrics):
    scores = ' '.join(f'{k} {metrics[k]:.4f}' for k in ('accuracy', 'macro_f1', 'balanced_accuracy', 'mAP', 'AP50'))
    return f'{scores} Δm {metrics["delta_m"]:+.2f}'


def validation_delta_m(metrics):
    """Mean relative change in validation macro-F1 and mAP, in percentage points."""
    return 50 * ((metrics['macro_f1'] - VAL_REFERENCE_F1) / VAL_REFERENCE_F1
                 + (metrics['mAP'] - VAL_REFERENCE_MAP) / VAL_REFERENCE_MAP)


def save_artifacts(run_dir, destination, filenames):
    """Copy completed run artifacts using atomic replacement of each destination file."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        temporary = destination / (name + '.tmp')
        shutil.copy2(run_dir / name, temporary)
        temporary.replace(destination / name)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    configuration = f'{args.backbone}_{args.init}_{args.adapt}'
    if args.unfreeze_last:
        configuration += f'_unfreeze{args.unfreeze_last}'
    run_dir = args.out / configuration
    if args.fold is not None:
        run_dir = args.out / f'fold_{args.fold}' / configuration
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / 'log.txt', 'w')
    log(json.dumps(vars(args), default=str), log_file)

    # Deterministic 90/10 split of the labeled course training set.
    rows = SARMultiTask(args.data, train=False, limit=args.limit)
    n = len(rows)
    if n < 2:
        raise ValueError(f"Need at least 2 labeled images for a train/validation split; found {n}.")
    if args.fold is None:
        permutation = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
        n_val = max(1, round(n * 0.10))
        val_indices = permutation[:n_val]
        train_indices = permutation[n_val:]
    else:
        from sarbench.cross_validation import stratified_folds
        train_indices, val_indices = stratified_folds(
            [int(row['label']) for row in rows.rows], args.folds, args.split_seed)[args.fold]
    split = {'fold': args.fold, 'folds': args.folds if args.fold is not None else None,
             'split_seed': args.split_seed if args.fold is not None else args.seed,
             'train_indices': train_indices, 'val_indices': val_indices,
             'train_image_ids': [rows.coco_images[rows.rows[i]['file_name']]['id'] for i in train_indices],
             'val_image_ids': [rows.coco_images[rows.rows[i]['file_name']]['id'] for i in val_indices]}
    (run_dir / 'split.json').write_text(json.dumps(split, indent=2))
    train_loader = make_loader(args, train_indices, train=True)
    val_loader = make_loader(args, val_indices, train=False)

    # Detection evaluation uses only annotations for this validation holdout.
    source_gt = json.loads((args.data / 'train' / 'instances.json').read_text())
    val_image_ids = {rows.coco_images[rows.rows[i]['file_name']]['id'] for i in val_indices}
    val_gt = {
        **source_gt,
        'images': [image for image in source_gt['images'] if image['id'] in val_image_ids],
        'annotations': [ann for ann in source_gt['annotations'] if ann['image_id'] in val_image_ids],
    }
    val_gt_path = run_dir / 'instances_val_holdout.json'
    val_gt_path.write_text(json.dumps(val_gt))
    log(f'data: {len(train_indices)} train, {len(val_indices)} validation images', log_file)

    # Model: adapters are created on the CPU, so they are added before .cuda()
    backbone = build_backbone(args.backbone,
                              pretrained=args.init == 'pretrained' and args.backbone_weights is None,
                              weights_path=args.backbone_weights)
    if args.adapt != 'full':
        add_adapters(backbone, args.adapt)  # also freezes the backbone
    unfreeze_last_blocks(backbone, args.unfreeze_last)
    model = MultiTaskModel(backbone, task_routing=args.adapt == 'moelora', anchor_profile=args.anchor_profile)
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'parameters: {parameters:,} total, {trainable:,} trainable', log_file)
    if args.unfreeze_last and (parameters > 130_000_000 or trainable > 40_000_000):
        raise ValueError('Partial fine-tuning exceeds the 130M total / 40M trainable parameter budget.')
    model.cuda()

    # Optimizer and schedule: trainable backbone parameters at --backbone-lr, all new modules at --lr
    groups = parameter_groups(model, args.backbone_lr, args.lr,
                              args.unfrozen_lr if args.unfreeze_last else None)
    log('optimizer groups: ' + ', '.join(f'{group["name"]} lr {group["lr"]:.2e}' for group in groups), log_file)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    warmup, total = len(train_loader), args.epochs * len(train_loader)  # one epoch of warmup
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total))

    # Train, validating every --eval-every epochs; the reported model is the last-epoch model
    history, start = [], time.time()
    best_val, best_detections = None, None
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, schedule, epoch, log_file, args.box_loss_weight)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val, val_detections = evaluate(model, val_loader, val_gt_path)
            val['delta_m'] = validation_delta_m(val)
            history.append({'epoch': epoch, **val})
            log(f'epoch {epoch} val: {headline(val)}', log_file)
            if args.select_best and (best_val is None or val['delta_m'] > best_val['delta_m']):
                best_val, best_detections = history[-1], val_detections
                torch.save(model.state_dict(), run_dir / 'best_model.pt')
    train_hours = (time.time() - start) / 3600

    # Save the trained model before final reporting.
    if args.select_best:
        (run_dir / 'best_model.pt').replace(run_dir / 'model.pt')
        val_detections = best_detections
    else:
        torch.save(model.state_dict(), run_dir / 'model.pt')
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log(f'training {train_hours:.2f} h, peak GPU memory {peak_gb:.1f} GB', log_file)

    summary = {'args': vars(args), 'split': split, 'parameters': parameters, 'trainable_parameters': trainable,
               'train_hours': train_hours, 'peak_gpu_memory_gb': peak_gb,
               'val_history': history, 'val': best_val if args.select_best else history[-1],
               'final_val': history[-1], 'selection': 'best_local_delta_m' if args.select_best else 'final_epoch',
               'test': None}
    (run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / 'predictions_val.json').write_text(json.dumps(val_detections))
    results_dir = args.results / run_dir.name / f'seed_{args.seed}'
    if args.fold is not None:
        results_dir = results_dir / f'fold_{args.fold}'
    save_artifacts(run_dir, results_dir, ('metrics.json', 'predictions_val.json', 'instances_val_holdout.json', 'split.json'))
    log(f'saved JSON results: {results_dir}', log_file)
    checkpoint_dir = args.checkpoints / run_dir.name / f'seed_{args.seed}'
    if args.fold is not None:
        checkpoint_dir = checkpoint_dir / f'fold_{args.fold}'
    save_artifacts(run_dir, checkpoint_dir, ('model.pt', 'metrics.json'))
    log(f'saved checkpoint: {checkpoint_dir}', log_file)


if __name__ == '__main__':
    main()
