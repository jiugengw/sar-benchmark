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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True, help='name for runs/<tag>, results/<tag> and the curves folder')
    parser.add_argument('--source', type=Path, default=CONTINUE_1 / 'weights/best.pt', help='detector to start from')
    parser.add_argument('--classifier', type=Path, default=CONTINUE_1 / 'image_classifier.pt',
                        help="image classifier used only for scoring (the source detector's)")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr0', type=float, default=0.001)
    parser.add_argument('--nwd-weight', type=float, default=0.0)
    parser.add_argument('--nwd-constant', type=float, default=16.0, help='typical object size in pixels')
    parser.add_argument('--speckle-prob', type=float, default=0.0)
    parser.add_argument('--speckle-looks', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=0, help='0 matches the control; change it to measure run-to-run noise')
    parser.add_argument('--all-data', action='store_true',
                        help='also train on the 939 holdout images (final-model recipe); the last epoch is used, and holdout '
                             'scores become optimistic because those images were trained on')
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
    run_root = ROOT / 'runs' / args.tag
    run_root.mkdir(parents=True, exist_ok=True)
    settings = {'source': str(args.source), 'classifier_for_scoring': str(args.classifier), 'epochs': args.epochs,
                'lr0': args.lr0, 'nwd_weight': args.nwd_weight, 'nwd_constant': args.nwd_constant,
                'speckle_prob': args.speckle_prob, 'speckle_looks': args.speckle_looks, 'seed': args.seed,
                'all_data': args.all_data,
                'control': 'continue_2 of job 329644: same recipe, no change (holdout mAP 44.98 full / 41.89 clean)'}
    write_json(run_root / 'experiment.json', settings)

    data_yaml = ROOT / 'data/yolo11s-sar-512/dataset.yaml'
    if args.all_data:  # train on the training split and the holdout together; validation still runs on the holdout
        base = (ROOT / 'data/yolo11s-sar-512').resolve()
        names = data_yaml.read_text().split('names:', 1)[1]
        data_yaml = run_root / 'dataset_all.yaml'
        data_yaml.write_text(f'path: {base}\ntrain:\n  - images/train\n  - images/val\nval: images/val\nnc: 9\nnames:{names}')
    started = time.monotonic()
    detector = YOLO(str(args.source))
    detector.train(  # the continue_2 recipe, unchanged
        data=str(data_yaml),
        epochs=args.epochs, imgsz=512, batch=16, workers=4, device=0,
        pretrained=True, optimizer='MuSGD', lr0=args.lr0, lrf=0.1, momentum=0.9, weight_decay=0.0005,
        warmup_epochs=1, warmup_bias_lr=0.0, cos_lr=True, seed=args.seed, deterministic=True,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0,
        translate=0.05, scale=0.2, fliplr=0.5, flipud=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
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

    for branch in ('one2many', 'one2one'):
        evaluate_export(weights, args.classifier, branch, ROOT / 'results' / args.tag / branch)
    from sarbench.twins import score
    for branch in ('one2many', 'one2one'):
        print(f'== {args.tag} {branch} (control continue_2 one2many: mAP 0.4498 full / 0.4189 clean)', flush=True)
        score(ROOT / 'results' / args.tag / branch, ROOT / 'data/course-data')


if __name__ == '__main__':
    main()
