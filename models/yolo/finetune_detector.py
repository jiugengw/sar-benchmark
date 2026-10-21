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


def enable_augment_annealing(detector, epochs: int, tail: int) -> None:
    """Shrink every augmentation linearly to zero over the run, holding the last `tail` epochs at zero (memorise).

    Jitter teaches finding objects, exact geometry teaches tight edges; instead of switching jitter off in one step,
    the scale/shift/flip/brightness strengths fall from their full value at epoch 0 to zero at epoch epochs - tail.
    The data loader is rebuilt each epoch so its worker processes see the new strengths."""
    from copy import copy

    def anneal(trainer):
        factor = max(0.0, 1.0 - trainer.epoch / max(1, epochs - tail))
        hyp = copy(trainer.args)
        for key in ('translate', 'scale', 'fliplr', 'flipud', 'hsv_v'):
            setattr(hyp, key, getattr(trainer.args, key) * factor)
        dataset = trainer.train_loader.dataset
        dataset.transforms = dataset.build_transforms(hyp)
        trainer.train_loader.reset()
        print(f'AUGMENT_ANNEALING epoch {trainer.epoch}: strength x{factor:.3f}', flush=True)

    detector.add_callback('on_train_epoch_start', anneal)


class BoxPaste:
    """Paste labelled objects from other training images of the SAME scene class into empty, similar-looking spots.

    Box-based copy-paste for SAR (cf. BoxPaste, Remote Sensing 2022), with the restrictions our data needs:
    - same scene class only: every image holds objects of exactly one class, so a turbine only goes into another
      wind-farm image (cross-scene pastes would create scenes that never occur);
    - only empty spots: the pasted rectangle (object plus a margin of its own background) overlaps no box;
    - background match: of `tries` random spots, the one whose surroundings best match the donor's margin in mean and
      spread of brightness is used, and the paste is skipped if even that one differs too much;
    - feathered border: the margin blends linearly into the new background, so there is no hard seam.
    The pasted box is the donor's label, unchanged. Training images only; added just before Ultralytics' Format step.
    """

    def __init__(self, dataset, p: float, max_objects: int, tries: int = 20):
        import numpy as np

        self.dataset, self.p, self.max_objects, self.tries = dataset, p, max_objects, tries
        self.by_class: dict[int, list[int]] = {}
        for index, label in enumerate(dataset.labels):
            classes = set(int(c) for c in np.asarray(label['cls']).ravel())
            if len(classes) == 1:
                self.by_class.setdefault(classes.pop(), []).append(index)

    def donor_objects(self, cls: int, rng):
        """A random image of this class (as the model sees it, 512 x 512) and its boxes in pixels (x1, y1, x2, y2)."""
        import numpy as np

        index = int(rng.choice(self.by_class[cls]))
        image = self.dataset.load_image(index)[0]
        h, w = image.shape[:2]
        xywh = np.asarray(self.dataset.labels[index]['bboxes'], dtype=np.float32).reshape(-1, 4)
        boxes = np.stack([(xywh[:, 0] - xywh[:, 2] / 2) * w, (xywh[:, 1] - xywh[:, 3] / 2) * h,
                          (xywh[:, 0] + xywh[:, 2] / 2) * w, (xywh[:, 1] + xywh[:, 3] / 2) * h], 1)
        return image, boxes

    def __call__(self, labels: dict) -> dict:
        import numpy as np
        from ultralytics.utils.instance import Instances

        rng = np.random
        cls = np.asarray(labels['cls']).ravel()
        if rng.random() >= self.p or len(cls) == 0 or int(cls[0]) not in self.by_class:
            return labels
        target_cls = int(cls[0])
        img = labels['img']
        h, w = img.shape[:2]
        instances = labels['instances']
        instances.convert_bbox('xyxy')
        if instances.normalized:
            instances.denormalize(w, h)
        occupied = [b for b in instances.bboxes]
        pasted, pasted_cls = [], []
        donor, donor_boxes = self.donor_objects(target_cls, rng)
        if donor.ndim == 2:
            donor = donor[..., None].repeat(img.shape[2], 2)
        for k in rng.permutation(len(donor_boxes))[: rng.randint(1, self.max_objects + 1)]:
            x1, y1, x2, y2 = donor_boxes[k]
            bw, bh = x2 - x1, y2 - y1
            if bw < 2 or bh < 2 or bw > w / 3 or bh > h / 3:
                continue
            margin = int(max(2, round(0.25 * max(bw, bh))))
            cx1, cy1 = int(max(0, np.floor(x1)) - margin), int(max(0, np.floor(y1)) - margin)
            cx2, cy2 = int(min(donor.shape[1], np.ceil(x2)) + margin), int(min(donor.shape[0], np.ceil(y2)) + margin)
            if cx1 < 0 or cy1 < 0 or cx2 > donor.shape[1] or cy2 > donor.shape[0]:
                continue  # object too close to its own image border for a full margin
            crop = donor[cy1:cy2, cx1:cx2].astype(np.float32)
            ch, cw = crop.shape[:2]
            if ch >= h or cw >= w:
                continue
            ring = np.ones((ch, cw), bool)
            ring[margin:ch - margin, margin:cw - margin] = False
            ring_mean, ring_std = crop[ring].mean(), crop[ring].std() + 1e-3
            best, best_cost = None, None
            for _ in range(self.tries):
                px, py = rng.randint(0, w - cw + 1), rng.randint(0, h - ch + 1)
                region = np.array([px, py, px + cw, py + ch], dtype=np.float32)
                if any(region[0] < b[2] and b[0] < region[2] and region[1] < b[3] and b[1] < region[3]
                       for b in occupied):
                    continue
                spot = img[py:py + ch, px:px + cw].astype(np.float32)
                if spot.std() < 2 or (spot == spot.flat[0]).mean() > 0.2:
                    continue  # flat letterbox padding from scale augmentation, not image content
                cost = abs(spot[ring].mean() - ring_mean) / ring_std + abs(spot[ring].std() - ring_std) / ring_std
                if best_cost is None or cost < best_cost:
                    best, best_cost = (px, py), cost
            if best is None or best_cost > 1.0:
                continue
            px, py = best
            ramp_y = np.minimum(np.arange(ch), np.arange(ch)[::-1])[:, None]
            ramp_x = np.minimum(np.arange(cw), np.arange(cw)[::-1])[None, :]
            alpha = np.clip(np.minimum(ramp_y, ramp_x) / margin, 0, 1)[..., None]
            spot = img[py:py + ch, px:px + cw].astype(np.float32)
            img[py:py + ch, px:px + cw] = np.clip(alpha * crop + (1 - alpha) * spot, 0, 255).astype(img.dtype)
            box = np.array([px + (x1 - cx1), py + (y1 - cy1), px + (x2 - cx1), py + (y2 - cy1)], dtype=np.float32)
            occupied.append(np.array([px, py, px + cw, py + ch], dtype=np.float32))
            pasted.append(box)
            pasted_cls.append(target_cls)
        if pasted:
            boxes = np.concatenate([instances.bboxes, np.stack(pasted)], 0).astype(np.float32)
            segments = instances.segments
            labels['instances'] = Instances(boxes, np.zeros((0, segments.shape[1] if segments.ndim == 3 else 0, 2),
                                                            dtype=np.float32), None, bbox_format='xyxy',
                                            normalized=False)
            labels['cls'] = np.concatenate([np.asarray(labels['cls']).reshape(-1, 1),
                                            np.array(pasted_cls, dtype=np.float32).reshape(-1, 1)], 0)
            labels['img'] = img
        return labels


def enable_box_paste(p: float, max_objects: int) -> None:
    """Insert BoxPaste into every training dataset's transforms, just before the final Format step."""
    from ultralytics.data import dataset as ul_dataset

    original = ul_dataset.YOLODataset.build_transforms

    def build_transforms(self, hyp=None):
        transforms = original(self, hyp)
        if self.augment:
            transforms.transforms.insert(len(transforms.transforms) - 1, BoxPaste(self, p, max_objects))
        return transforms

    ul_dataset.YOLODataset.build_transforms = build_transforms
    print(f'BoxPaste: probability {p}, up to {max_objects} objects, same scene class, matched background', flush=True)


class Rotate90:
    """Rotate the image and its boxes by 90, 180 or 270 degrees, with probability p (each angle equally likely).

    Exact for axis-aligned boxes: a box's width and height just swap, so no box grows (unlike Ultralytics' `degrees`,
    which enlarges every box to hold the rotated corners). Radar objects such as bridges, harbours and runways appear
    at any heading, so each one is seen at four orientations. Square images only (all of ours after the letterbox)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, labels: dict) -> dict:
        import numpy as np
        from ultralytics.utils.instance import Instances

        img = labels['img']
        h, w = img.shape[:2]
        if np.random.random() >= self.p or h != w:
            return labels
        k = np.random.randint(1, 4)
        instances = labels['instances']
        instances.convert_bbox('xyxy')
        if instances.normalized:
            instances.denormalize(w, h)
        boxes = instances.bboxes.copy()
        for _ in range(k):  # np.rot90 turns counter-clockwise: (x, y) -> (y, w - x)
            boxes = np.stack([boxes[:, 1], w - boxes[:, 2], boxes[:, 3], w - boxes[:, 0]], 1)
        labels['img'] = np.ascontiguousarray(np.rot90(img, k))
        segments = instances.segments
        labels['instances'] = Instances(boxes.astype(np.float32),
                                        np.zeros((0, segments.shape[1] if segments.ndim == 3 else 0, 2), dtype=np.float32),
                                        None, bbox_format='xyxy', normalized=False)
        return labels


def enable_rotate90(p: float) -> None:
    """Insert Rotate90 into every training dataset's transforms, just before the final Format step."""
    from ultralytics.data import dataset as ul_dataset

    original = ul_dataset.YOLODataset.build_transforms

    def build_transforms(self, hyp=None):
        transforms = original(self, hyp)
        if self.augment:  # training only; validation images stay untouched
            transforms.transforms.insert(len(transforms.transforms) - 1, Rotate90(p))
        return transforms

    ul_dataset.YOLODataset.build_transforms = build_transforms
    print(f'Rotate90: probability {p} of a 90, 180 or 270 degree turn', flush=True)


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


def repeat_labels(labels: list[dict], factors: dict[int, int], only_width: int | None = None) -> list[dict]:
    """Repeat-factor sampling: list every image containing one of the classes in `factors` that many times instead of
    once (the largest factor among its classes).

    only_width: repeat only images whose ORIGINAL width is this (data/yolo11s-sar-512/image_map.json; every training
    image is stored at 512, so 256 px sources are told apart there). 512 picks the M4-SAR-like source of bridge,
    harbor and oil tank, whose objects are still data-limited, and skips their well-learned 256 px source."""
    widths = {}
    if only_width is not None:
        image_map = json.loads((ROOT / 'data/yolo11s-sar-512/image_map.json').read_text())
        widths = {key.split('/')[-1]: info['width'] for key, info in image_map.items()}
    out = []
    for label in labels:
        out.append(label)
        classes = {int(c) for c in label['cls'].ravel()} if len(label['cls']) else set()
        factor = max((factors[c] for c in classes & factors.keys()), default=1)
        if factor > 1 and (only_width is None or widths.get(Path(label['im_file']).name) == only_width):
            out.extend(dict(label) for _ in range(factor - 1))
    return out


def enable_class_repeat(factors: dict[int, int], only_width: int | None = None) -> None:
    """Apply repeat_labels to every training dataset (validation keeps each image once)."""
    from ultralytics.data import dataset as ul_dataset

    original = ul_dataset.YOLODataset.get_labels

    def get_labels(self):
        labels = original(self)
        if not self.augment:
            return labels
        labels = repeat_labels(labels, factors, only_width)
        self.im_files = [label['im_file'] for label in labels]
        print(f'class repeat: {len(labels)} training entries after repeating {factors}', flush=True)
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


def enable_alpha_iou(alpha: float) -> None:
    """alpha-CIoU box loss (He et al., NeurIPS 2021): 1 - IoU^a + penalty^a instead of 1 - IoU + penalty.

    With a > 1 the gradient grows for boxes that already overlap well, so training pushes harder on exactly the
    0.75-0.95 range the strict thresholds score; the paper also reports robustness to noisy boxes. Only the box loss
    changes (Ultralytics' loss module imports bbox_iou by name; the label assigner keeps its own copy)."""
    from ultralytics.utils import loss as ul_loss
    from ultralytics.utils.metrics import bbox_iou

    def alpha_bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        if not CIoU:
            return bbox_iou(box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, eps=eps)
        iou = bbox_iou(box1, box2, xywh=xywh, eps=eps)
        penalty = (iou - bbox_iou(box1, box2, xywh=xywh, CIoU=True, eps=eps)).clamp(min=0)  # centre distance + shape
        return iou.clamp(min=eps).pow(alpha) - penalty.clamp(min=eps).pow(alpha)

    ul_loss.bbox_iou = alpha_bbox_iou
    print(f'alpha-CIoU box loss, alpha {alpha}', flush=True)


def tune_assigner(topk: int | None, beta: float | None) -> None:
    """Change YOLO's task-aligned assigner: how many grid points learn each object (top-k of the one-to-many head,
    default 10) and how strongly box overlap weighs in choosing them (beta, default 6; higher favours points that
    already predict tight boxes)."""
    from ultralytics.utils import loss as ul_loss

    original_init = ul_loss.v8DetectionLoss.__init__

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        original_init(self, model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if topk is not None and tal_topk == 10:  # the one-to-many head; the one-to-one head (7 -> 1) is left alone
            self.assigner.topk = self.assigner.topk2 = topk
        if beta is not None:
            self.assigner.beta = beta

    ul_loss.v8DetectionLoss.__init__ = __init__
    print(f'Assigner: one-to-many top-k {topk or 10}, beta {beta or 6.0}', flush=True)


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
                        help='comma-separated classes whose training images are shown --repeat-factor times; '
                             'name:N sets that class\'s own factor, e.g. airport,harbor,bridge:4,playground:4')
    parser.add_argument('--repeat-factor', type=int, default=2)
    parser.add_argument('--repeat-width', type=int, help='repeat only images whose original width is this (512 = the '
                                                         'M4-SAR-like source); default: every matching image')
    parser.add_argument('--seed', type=int, default=0, help='0 matches the control; change it to measure run-to-run noise')
    parser.add_argument('--no-augment', action='store_true',
                        help='memorise stage: no flips, shifts, scaling or brightness changes, so the detector learns to '
                             'reproduce the exact training boxes of every labelled scene (official near-copies benefit)')
    parser.add_argument('--missing-label-classes', default='',
                        help='comma-separated classes with many missing labels (e.g. bridge,playground): confident '
                             'predictions away from every label are not punished (Background Recalibration Loss idea)')
    parser.add_argument('--missing-label-threshold', type=float, default=0.5)
    parser.add_argument('--box-gain', type=float, default=7.5,
                        help='weight of the box loss (Ultralytics default 7.5); higher favours exact box edges')
    parser.add_argument('--keep-flips', action='store_true',
                        help='with --no-augment: keep the flips, which leave box geometry exact, and switch off only '
                             'the shifts, scaling and brightness changes that blur it')
    parser.add_argument('--all-data', action='store_true',
                        help='also train on the 939 holdout images (final-model recipe); the last epoch is used, and holdout '
                             'scores become optimistic because those images were trained on')
    parser.add_argument('--constant-lr', action='store_true',
                        help='hold the learning rate at --lr0 (no cosine decay); for weight averaging (SWA) checkpoints')
    parser.add_argument('--save-period', type=int, default=10, help='save a checkpoint every N epochs')
    parser.add_argument('--aug-scale', type=float, default=0.2, help='scale jitter (+-) when augmenting')
    parser.add_argument('--aug-translate', type=float, default=0.05, help='shift jitter (share of the image)')
    parser.add_argument('--mosaic', type=float, default=0.0, help='probability of 4-image mosaics when augmenting')
    parser.add_argument('--photometric', action='store_true',
                        help='with --no-augment: keep brightness jitter and allow --speckle-prob; they change how the '
                             'image looks, not where objects are, so boxes stay pixel-exact')
    parser.add_argument('--anneal-augment', type=int, metavar='TAIL',
                        help='shrink augmentation linearly to zero, the last TAIL epochs without any (see '
                             'enable_augment_annealing)')
    parser.add_argument('--stem-stride', type=int, choices=(1, 2), default=2,
                        help='1: the first convolution keeps full resolution, so every level works at twice the detail '
                             '(strides 2/4/8/16 instead of 4/8/16/32) on the same 512 input; about 4x the activation '
                             'memory, so lower --batch')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--alpha-iou', type=float, help='alpha-CIoU box loss with this power (the paper uses 3)')
    parser.add_argument('--tal-topk', type=int, help="grid points per object in the one-to-many head (YOLO26: 10)")
    parser.add_argument('--tal-beta', type=float, help='weight of box overlap when picking them (YOLO26: 6)')
    parser.add_argument('--box-paste', type=float, default=0.0,
                        help='probability of pasting objects from another same-class training image (BoxPaste)')
    parser.add_argument('--box-paste-max', type=int, default=3, help='at most this many pasted objects per image')
    parser.add_argument('--rotate90', type=float, default=0.0,
                        help='probability of turning a training image and its boxes by 90, 180 or 270 degrees')
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
    if args.box_paste:
        if args.no_augment:
            raise ValueError('--box-paste is an augmentation; memorise (--no-augment) runs without it')
        enable_box_paste(args.box_paste, args.box_paste_max)
    if args.rotate90:
        if args.no_augment:
            raise ValueError('--rotate90 is an augmentation; memorise (--no-augment) runs without it')
        enable_rotate90(args.rotate90)
    if args.alpha_iou:
        enable_alpha_iou(args.alpha_iou)
    if args.tal_topk or args.tal_beta:
        tune_assigner(args.tal_topk, args.tal_beta)
    if args.speckle_prob:
        enable_speckle(args.speckle_prob, args.speckle_looks)
    missing = [name.strip() for name in args.missing_label_classes.split(',') if name.strip()]
    if missing:
        if set(missing) - set(CLASSES):
            raise ValueError(f'unknown classes {set(missing) - set(CLASSES)}; choose from {CLASSES}')
        enable_missing_label_loss({CLASSES.index(name) for name in missing}, args.missing_label_threshold)
    repeat = {}
    for entry in (e.strip() for e in args.repeat_classes.split(',') if e.strip()):
        name, _, factor = entry.partition(':')
        if name not in CLASSES:
            raise ValueError(f'unknown class {name}; choose from {CLASSES}')
        repeat[CLASSES.index(name)] = int(factor) if factor else args.repeat_factor
    if repeat:
        enable_class_repeat(repeat, args.repeat_width)
    run_root = ROOT / 'runs' / args.tag
    run_root.mkdir(parents=True, exist_ok=True)
    settings = {'source': str(args.source), 'classifier_for_scoring': str(args.classifier), 'epochs': args.epochs,
                'lr0': args.lr0, 'nwd_weight': args.nwd_weight, 'nwd_constant': args.nwd_constant,
                'speckle_prob': args.speckle_prob, 'speckle_looks': args.speckle_looks, 'seed': args.seed,
                'repeat_classes': args.repeat_classes, 'repeat_factor': args.repeat_factor,
                'repeat_width': args.repeat_width,
                'all_data': args.all_data, 'data': str(args.data), 'no_augment': args.no_augment, 'keep_flips': args.keep_flips, 'missing_label_classes': args.missing_label_classes, 'box_gain': args.box_gain,
                'constant_lr': args.constant_lr, 'aug_scale': args.aug_scale, 'aug_translate': args.aug_translate,
                'mosaic': args.mosaic, 'photometric': args.photometric, 'stem_stride': args.stem_stride, 'alpha_iou': args.alpha_iou,
                'tal_topk': args.tal_topk, 'tal_beta': args.tal_beta,
                'box_paste': args.box_paste, 'box_paste_max': args.box_paste_max, 'rotate90': args.rotate90,
                'batch': args.batch, 'anneal_augment': args.anneal_augment,
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
    augmentation = dict(translate=args.aug_translate, scale=args.aug_scale, fliplr=0.5, flipud=0.5, hsv_v=0.2,
                        mosaic=args.mosaic)
    if args.no_augment:
        if args.speckle_prob and not args.photometric:
            raise ValueError('--no-augment and --speckle-prob contradict each other (unless --photometric)')
        augmentation = {key: 0.0 for key in augmentation}
        if args.photometric:
            augmentation.update(hsv_v=0.2)
        if args.keep_flips:  # a mirrored box is still pixel-exact
            augmentation.update(fliplr=0.5, flipud=0.5)
    elif args.keep_flips or args.photometric:
        raise ValueError('--keep-flips and --photometric only apply with --no-augment')
    started = time.monotonic()
    detector = YOLO(str(args.source))
    if args.stem_stride != 2:
        # The trainer rebuilds the model from its YAML and loads the weights into it, so change the YAML: the first
        # Conv's stride. No weight changes shape; the model's strides are recomputed when it is rebuilt.
        stem = detector.model.yaml['backbone'][0]
        if stem[2] != 'Conv' or stem[3][2] != 2:
            raise RuntimeError(f'Unexpected first layer {stem}')
        stem[3][2] = args.stem_stride
    if args.anneal_augment is not None:
        if args.no_augment:
            raise ValueError('--anneal-augment needs augmentation to anneal; drop --no-augment')
        enable_augment_annealing(detector, args.epochs, args.anneal_augment)
    detector.train(  # the continue_2 recipe, unchanged
        data=str(data_yaml),
        epochs=args.epochs, imgsz=512, batch=args.batch, workers=4, device=0,
        pretrained=True, optimizer='MuSGD', lr0=args.lr0, lrf=1.0 if args.constant_lr else 0.1, momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=0 if args.constant_lr else 1, warmup_bias_lr=0.0, cos_lr=not args.constant_lr,
        seed=args.seed, deterministic=True, mixup=0.0, copy_paste=0.0, close_mosaic=0,
        **augmentation, hsv_h=0.0, hsv_s=0.0, box=args.box_gain,
        multi_scale=0.0, amp=True, patience=args.epochs + 1,
        project=str(run_root), name='train', plots=True, save_period=args.save_period,
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
