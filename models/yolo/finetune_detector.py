"""Fine-tune a trained YOLO26 detector for 40 more epochs, changing one thing, against a known control.

The default recipe copies the second continuation block (continue_2 of job 329644) exactly: start from the
continue_1 detector, 40 epochs, MuSGD, learning rate 0.001 with cosine decay, no mosaic, flips. So continue_2
(holdout mAP 44.98% full / 41.89% clean, one-to-many head) is the control, and each option below changes one thing:

    --nwd-weight 0.5     box loss = (1 - w) x CIoU loss + w x NWD loss (normalized Wasserstein distance, in pixels)
    --speckle-prob 0.5   multiply training images by random gamma "speckle" noise with this probability

After training, both prediction heads are scored on the full and clean holdout through the exact submission code.
They use the source detector's image classifier, so judge detection mAP; refit the classifier only for a winner.
Run from the repository root on a GPU, e.g. python models/yolo/finetune_detector.py --tag ft40-nwd --nwd-weight 0.5
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONTINUE_1 = ROOT / 'runs/deadline-yolo26m-329644/continue_1'


def enable_nwd(weight: float, constant: float = 16.0) -> None:
    """Blend an NWD term into Ultralytics' box loss: loss_box = (1 - weight) * CIoU + weight * NWD.

    NWD models each box as a 2-D Gaussian (centre, half width and height) and measures the Wasserstein distance
    between two of them in pixels: exp(-sqrt(W2) / constant). Unlike IoU it changes smoothly for tiny boxes that
    barely overlap. The constant is roughly the typical object size (about 16 px in our 512 px images)."""
    import torch
    from ultralytics.utils import loss as ul_loss

    original = ul_loss.BboxLoss.forward

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
                fg_mask, imgsz, stride):
        loss_iou, loss_dfl = original(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                      target_scores_sum, fg_mask, imgsz, stride)
        fg = fg_mask.nonzero(as_tuple=True)
        if fg[0].numel() == 0:
            return loss_iou, loss_dfl
        scale = stride[fg[1]]  # boxes arrive in grid units; stride converts them to pixels
        pred = pred_bboxes[fg] * scale
        target = target_bboxes[fg] * scale
        nwd = normalized_wasserstein(pred, target, constant)
        w = target_scores[fg].sum(-1, keepdim=True)
        loss_nwd = ((1.0 - nwd).unsqueeze(-1) * w).sum() / target_scores_sum
        return (1.0 - weight) * loss_iou + weight * loss_nwd, loss_dfl

    ul_loss.BboxLoss.forward = forward


def normalized_wasserstein(a, b, constant: float):
    """NWD between xyxy boxes a and b (same shape [..., 4]): exp(-sqrt(W2) / constant)."""
    import torch

    ca, cb = (a[..., :2] + a[..., 2:]) / 2, (b[..., :2] + b[..., 2:]) / 2
    ha, hb = (a[..., 2:] - a[..., :2]) / 2, (b[..., 2:] - b[..., :2]) / 2
    w2 = ((ca - cb) ** 2).sum(-1) + ((ha - hb) ** 2).sum(-1)
    return torch.exp(-torch.sqrt(w2 + 1e-7) / constant)


class SpeckleNoise:
    """Multiply the image by gamma-distributed noise with mean 1 (SAR-like speckle), with probability p.

    'looks' sets the strength: a gamma(L, 1/L) multiplier has standard deviation 1/sqrt(L), so L = 4 gives about
    50% grain. The same noise is used for every channel because the images are grayscale."""

    def __init__(self, p: float = 0.5, looks: float = 4.0):
        self.p, self.looks = p, looks  # NumPy's global generator: Ultralytics seeds it per data-loader worker

    def __call__(self, labels: dict) -> dict:
        import numpy as np

        if np.random.random() >= self.p:
            return labels
        img = labels['img']
        noise = np.random.gamma(self.looks, 1.0 / self.looks, size=img.shape[:2]).astype(np.float32)
        noisy = img.astype(np.float32) * (noise[..., None] if img.ndim == 3 else noise)
        labels['img'] = np.clip(noisy, 0, 255).astype(img.dtype)
        return labels


def enable_speckle(p: float, looks: float) -> None:
    """Insert SpeckleNoise into every training dataset's transforms, just before the final Format step."""
    from ultralytics.data import dataset as ul_dataset

    original = ul_dataset.YOLODataset.build_transforms

    def build_transforms(self, hyp=None):
        transforms = original(self, hyp)
        if self.augment:  # training only; validation images stay untouched
            transforms.transforms.insert(len(transforms.transforms) - 1, SpeckleNoise(p, looks))
        return transforms

    ul_dataset.YOLODataset.build_transforms = build_transforms


CLASSES = ('aircraft', 'airport', 'bridge', 'car', 'harbor', 'oil_tank', 'playground', 'ship', 'wind_turbine')


def repeat_labels(labels: list[dict], class_ids: set[int], factor: int) -> list[dict]:
    """Repeat-factor sampling: list every image containing one of class_ids `factor` times instead of once."""
    out = []
    for label in labels:
        out.append(label)
        classes = {int(c) for c in label['cls'].ravel()} if len(label['cls']) else set()
        if classes & class_ids:
            out.extend(dict(label) for _ in range(factor - 1))
    return out


def enable_class_repeat(class_ids: set[int], factor: int) -> None:
    """Apply repeat_labels to every training dataset (validation keeps each image once)."""
    from ultralytics.data import dataset as ul_dataset

    original = ul_dataset.YOLODataset.get_labels

    def get_labels(self):
        labels = original(self)
        if not self.augment:
            return labels
        labels = repeat_labels(labels, class_ids, factor)
        self.im_files = [label['im_file'] for label in labels]
        print(f'class repeat: {len(labels)} training entries after repeating {sorted(class_ids)} x{factor}', flush=True)
        return labels

    ul_dataset.YOLODataset.get_labels = get_labels


def near_labels(anchor_points, gt_bboxes, mask_gt, factor: float = 2.0):
    """(batch, anchors) True where an anchor centre lies inside some label box enlarged `factor` times about its centre.

    anchor_points (anchors, 2) and gt_bboxes (batch, max_boxes, 4, xyxy) are in pixels; mask_gt (batch, max_boxes, 1)
    marks real boxes among the padding."""
    import torch

    centre = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2
    half = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]) * factor / 2
    lo, hi = (centre - half)[:, :, None], (centre + half)[:, :, None]  # (b, boxes, 1, 2)
    points = anchor_points[None, None]  # (1, 1, anchors, 2)
    inside = ((points >= lo) & (points <= hi)).all(-1) & mask_gt.bool()  # (b, boxes, anchors)
    return inside.any(1) if inside.shape[1] else torch.zeros(inside.shape[0], anchor_points.shape[0], dtype=torch.bool,
                                                             device=anchor_points.device)


class RecalibratedBCE:
    """BCE classification loss that does not punish a probable missing label (Background Recalibration Loss idea).

    For the chosen classes, a spot keeps zero loss when it is background in the targets (no label assigned), the
    model already scores it above `threshold`, and it lies outside the zone around every label (so the one-to-one
    head still learns to suppress duplicates next to labelled objects). Everything else is the usual BCE."""

    def __init__(self, class_ids, threshold: float, owner):
        self.class_ids, self.threshold, self.owner = sorted(class_ids), threshold, owner

    def __call__(self, pred, target):
        import torch
        import torch.nn.functional as F

        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        near = getattr(self.owner, '_near_labels', None)
        if near is None or near.shape != pred.shape[:2]:
            return loss
        with torch.no_grad():
            skip = torch.zeros_like(loss, dtype=torch.bool)
            cols = list(self.class_ids)
            skip[..., cols] = (target[..., cols] == 0) & (pred[..., cols].sigmoid() > self.threshold) & ~near[..., None]
            self.owner._skipped = int(skip.sum())
        return loss.masked_fill(skip, 0.0)


def enable_missing_label_loss(class_ids: set[int], threshold: float = 0.5, factor: float = 2.0) -> None:
    """Install RecalibratedBCE in every detection loss Ultralytics builds (both YOLO26 heads use v8DetectionLoss)."""
    from ultralytics.utils import loss as ul_loss

    original_init = ul_loss.v8DetectionLoss.__init__
    if getattr(original_init, 'missing_label_loss', False):
        return

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        assign = self.assigner

        def assigner(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
            self._near_labels = near_labels(anc_points, gt_bboxes, mask_gt, factor)  # anchors and boxes in pixels
            return assign(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

        self.assigner = assigner
        self.bce = RecalibratedBCE(class_ids, threshold, self)

    __init__.missing_label_loss = True
    ul_loss.v8DetectionLoss.__init__ = __init__
    print(f'Missing-label loss: no penalty for confident unlabelled {sorted(CLASSES[c] for c in class_ids)} '
          f'(score > {threshold}, outside {factor}x every label)', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True, help='name for runs/<tag>, results/<tag> and the curves folder')
    parser.add_argument('--source', type=Path, default=CONTINUE_1 / 'weights/best.pt', help='detector to start from')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/yolo11s-sar-512/dataset.yaml',
                        help='dataset YAML; a variant from build_variant_dataset.py for the context-box or filter experiments')
    parser.add_argument('--classifier', type=Path, default=CONTINUE_1 / 'image_classifier.pt',
                        help="image classifier used only for scoring (the source detector's)")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr0', type=float, default=0.001)
    parser.add_argument('--nwd-weight', type=float, default=0.0)
    parser.add_argument('--nwd-constant', type=float, default=16.0, help='typical object size in pixels')
    parser.add_argument('--speckle-prob', type=float, default=0.0)
    parser.add_argument('--speckle-looks', type=float, default=4.0)
    parser.add_argument('--repeat-classes', default='',
                        help='comma-separated classes whose training images are shown --repeat-factor times, '
                             'e.g. bridge,playground,wind_turbine')
    parser.add_argument('--repeat-factor', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0, help='0 matches the control; change it to measure run-to-run noise')
    parser.add_argument('--no-augment', action='store_true',
                        help='memorise stage: no flips, shifts, scaling or brightness changes, so the detector learns to '
                             'reproduce the exact training boxes of every labelled scene (official near-copies benefit)')
    parser.add_argument('--missing-label-classes', default='',
                        help='comma-separated classes with many missing labels (e.g. bridge,playground): confident '
                             'predictions away from every label are not punished (Background Recalibration Loss idea)')
    parser.add_argument('--missing-label-threshold', type=float, default=0.5)
    parser.add_argument('--keep-flips', action='store_true',
                        help='with --no-augment: keep the flips, which leave box geometry exact, and switch off only '
                             'the shifts, scaling and brightness changes that blur it')
    parser.add_argument('--all-data', action='store_true',
                        help='also train on the 939 holdout images (final-model recipe); the last epoch is used, and holdout '
                             'scores become optimistic because those images were trained on')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='save detector and curves only; use when a later stage refits the image classifier')
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO
    from models.yolo.deadline_recovery import evaluate_export, write_json

    if not torch.cuda.is_available():
        raise RuntimeError('An allocated CUDA GPU is required')
    for path in (args.source, args.classifier):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.nwd_weight:
        enable_nwd(args.nwd_weight, args.nwd_constant)
    if args.speckle_prob:
        enable_speckle(args.speckle_prob, args.speckle_looks)
    missing = [name.strip() for name in args.missing_label_classes.split(',') if name.strip()]
    if missing:
        if set(missing) - set(CLASSES):
            raise ValueError(f'unknown classes {set(missing) - set(CLASSES)}; choose from {CLASSES}')
        enable_missing_label_loss({CLASSES.index(name) for name in missing}, args.missing_label_threshold)
    repeat = [name.strip() for name in args.repeat_classes.split(',') if name.strip()]
    if repeat:
        unknown = set(repeat) - set(CLASSES)
        if unknown:
            raise ValueError(f'unknown classes {unknown}; choose from {CLASSES}')
        enable_class_repeat({CLASSES.index(name) for name in repeat}, args.repeat_factor)
    run_root = ROOT / 'runs' / args.tag
    run_root.mkdir(parents=True, exist_ok=True)
    settings = {'source': str(args.source), 'classifier_for_scoring': str(args.classifier), 'epochs': args.epochs,
                'lr0': args.lr0, 'nwd_weight': args.nwd_weight, 'nwd_constant': args.nwd_constant,
                'speckle_prob': args.speckle_prob, 'speckle_looks': args.speckle_looks, 'seed': args.seed,
                'repeat_classes': args.repeat_classes, 'repeat_factor': args.repeat_factor,
                'all_data': args.all_data, 'data': str(args.data), 'no_augment': args.no_augment, 'keep_flips': args.keep_flips, 'missing_label_classes': args.missing_label_classes,
                'skip_evaluation': args.skip_evaluation,
                'control': 'continue_2 of job 329644: same recipe, no change (holdout mAP 44.98 full / 41.89 clean)'}
    write_json(run_root / 'experiment.json', settings)

    data_yaml = args.data.resolve()
    if args.all_data:  # train on the training split and the holdout together; validation still runs on the holdout
        base = data_yaml.parent
        names = data_yaml.read_text().split('names:', 1)[1]
        data_yaml = run_root / 'dataset_all.yaml'
        data_yaml.write_text(f'path: {base}\ntrain:\n  - images/train\n  - images/val\nval: images/val\nnc: 9\nnames:{names}')
    from models.yolo.input_filters import install_training_filter
    install_training_filter()  # SAR_INPUT_FILTER, if set
    augmentation = dict(translate=0.05, scale=0.2, fliplr=0.5, flipud=0.5, hsv_v=0.2)
    if args.no_augment:
        if args.speckle_prob:
            raise ValueError('--no-augment and --speckle-prob contradict each other')
        augmentation = {key: 0.0 for key in augmentation}
        if args.keep_flips:  # a mirrored box is still pixel-exact
            augmentation.update(fliplr=0.5, flipud=0.5)
    elif args.keep_flips:
        raise ValueError('--keep-flips only applies with --no-augment')
    started = time.monotonic()
    detector = YOLO(str(args.source))
    detector.train(  # the continue_2 recipe, unchanged
        data=str(data_yaml),
        epochs=args.epochs, imgsz=512, batch=16, workers=4, device=0,
        pretrained=True, optimizer='MuSGD', lr0=args.lr0, lrf=0.1, momentum=0.9, weight_decay=0.0005,
        warmup_epochs=1, warmup_bias_lr=0.0, cos_lr=True, seed=args.seed, deterministic=True,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0,
        **augmentation, hsv_h=0.0, hsv_s=0.0,
        multi_scale=0.0, amp=True, patience=args.epochs + 1,
        project=str(run_root), name='train', plots=True, save_period=10,
    )
    weights = run_root / ('train/weights/last.pt' if args.all_data else 'train/weights/best.pt')
    write_json(run_root / 'compute.json', {'wall_seconds': time.monotonic() - started})

    curves = ROOT / 'results/training-curves' / args.tag  # runs/ is not committed; keep the history
    curves.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_root / 'train/results.csv', curves / 'training.csv')
    shutil.copy2(run_root / 'train/args.yaml', curves / 'args.yaml')
    write_json(curves / 'experiment.json', settings)

    if args.skip_evaluation:
        print(f'DETECTOR_READY {weights}; image-classifier refit and evaluation deferred', flush=True)
        return

    for branch in ('one2many', 'one2one'):
        evaluate_export(weights, args.classifier, branch, ROOT / 'results' / args.tag / branch)
    from sarbench.twins import score
    for branch in ('one2many', 'one2one'):
        print(f'== {args.tag} {branch} (control continue_2 one2many: mAP 0.4498 full / 0.4189 clean)', flush=True)
        score(ROOT / 'results' / args.tag / branch, ROOT / 'data/course-data')


if __name__ == '__main__':
    main()
