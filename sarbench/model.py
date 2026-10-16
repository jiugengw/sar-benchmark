"""The multi-task model: one shared ViT backbone feeds a classification head and a Faster R-CNN detector."""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .adapters import set_task


class VariableAnchorRPNHead(nn.Module):
    """Share the RPN tower; prediction layers depend on anchors per location."""

    def __init__(self, channels, counts):
        super().__init__()
        self.counts = counts
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU())
        self.cls_logits = nn.ModuleDict({str(n): nn.Conv2d(channels, n, 1) for n in set(counts)})
        self.bbox_pred = nn.ModuleDict({str(n): nn.Conv2d(channels, n * 4, 1) for n in set(counts)})
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, features):
        logits, boxes = [], []
        if len(features) != len(self.counts):
            raise ValueError('RPN feature levels do not match anchor configuration')
        for feature, count in zip(features, self.counts):
            hidden = self.conv(feature)
            logits.append(self.cls_logits[str(count)](hidden))
            boxes.append(self.bbox_pred[str(count)](hidden))
        return logits, boxes


class SimpleFeaturePyramid(nn.Module):
    """ViTDet neck (Li et al., 2022): the single stride-16 ViT map -> 256-channel maps at strides 4, 8, 16, 32, 64."""

    def __init__(self, dim=768, channels=256):
        super().__init__()
        self.rescale = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 2, stride=2), nn.GroupNorm(32, dim // 2), nn.GELU(),
                          nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2)),  # stride 4
            nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),  # stride 8
            nn.Identity(),  # stride 16
            nn.MaxPool2d(2),  # stride 32
        ])
        self.project = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, channels, 1, bias=False), nn.GroupNorm(32, channels),
                          nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(32, channels))
            for c in (dim // 4, dim // 2, dim, dim))

    def forward(self, x):
        levels = [project(rescale(x)) for rescale, project in zip(self.rescale, self.project)]
        levels.append(F.max_pool2d(levels[-1], 1, stride=2))  # stride 64, as torchvision's FPN LastLevelMaxPool
        return OrderedDict(zip(['0', '1', '2', '3', 'pool'], levels))


class Stride4DetailBranch(nn.Module):
    """Encode input pixels directly at stride 4 and fuse them with the ViT pyramid."""

    def __init__(self, channels=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.GroupNorm(8, 64), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(channels + 64, channels, 1, bias=False),
                                  nn.GroupNorm(32, channels), nn.GELU(),
                                  nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.GroupNorm(32, channels))

    def forward(self, image, pyramid):
        details = self.stem(image)
        base = pyramid['0']
        if details.shape[-2:] != base.shape[-2:]:
            details = F.interpolate(details, size=base.shape[-2:], mode='bilinear', align_corners=False)
        pyramid['0'] = base + self.fuse(torch.cat((base, details), dim=1))
        return pyramid


class MultiTaskModel(nn.Module):
    """backbone tokens -> classification logits (mean token -> LayerNorm -> Linear) and Faster R-CNN detections."""

    def __init__(self, backbone, task_routing: bool, num_classes=9, anchor_profile='default',
                 detail_branch=False):
        super().__init__()
        self.backbone = backbone
        self.task_routing = task_routing  # MoE-LoRA: the backbone runs once per task, with that task's routers
        self.cls_head = nn.Sequential(nn.LayerNorm(backbone.embed_dim), nn.Linear(backbone.embed_dim, num_classes))
        self.spatial_backbone = hasattr(backbone, 'feature_channels')
        self.neck = (FeaturePyramidNetwork(backbone.feature_channels, 256, extra_blocks=LastLevelMaxPool())
                     if self.spatial_backbone else SimpleFeaturePyramid(backbone.embed_dim))
        self.detail_branch = Stride4DetailBranch() if detail_branch and not self.spatial_backbone else None
        if detail_branch and self.spatial_backbone:
            raise ValueError('The stride-4 detail branch is currently for transformer backbones only.')
        # Small objects (median ~16 px at 512 px, 99 % below ~240 px): two anchor scales per octave, from 8 px at
        # stride 4 up to 181 px at stride 64, times 3 aspect ratios = 6 anchors per location.
        sizes = ((8, 11), (16, 23), (32, 45), (64, 91), (128, 181))
        ratios = (0.5, 1.0, 2.0)
        if anchor_profile == 'wide':
            # Isolate shape coverage: preserve scales, add elongated anchors.
            ratios = (0.2, 0.5, 1.0, 2.0, 5.0)
        elif anchor_profile == 'large':
            # Isolate scale coverage: extend the coarsest level to larger objects.
            sizes = (*sizes[:4], (181, 256))
        elif anchor_profile == 'small_extreme':
            sizes = ((6, 8, 11), *sizes[1:])
            ratios = (0.25, 0.5, 1.0, 2.0, 4.0)
        elif anchor_profile != 'default':
            raise ValueError(f'Unknown anchor profile: {anchor_profile}')
        anchors = AnchorGenerator(sizes, (ratios,) * 5)
        self.anchor_configuration = {'sizes': sizes, 'aspect_ratios': (ratios,) * 5,
                                     'strides': (4, 8, 16, 32, 64)}
        counts = anchors.num_anchors_per_location()
        rpn_head = (VariableAnchorRPNHead(256, counts) if len(set(counts)) > 1
                    else RPNHead(256, counts[0]))
        # Everything else is torchvision's Faster R-CNN default.
        self.rpn = RegionProposalNetwork(
            anchors, rpn_head, fg_iou_thresh=0.7, bg_iou_thresh=0.3, batch_size_per_image=256,
            positive_fraction=0.5, pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=2000, testing=1000), nms_thresh=0.7)
        self.roi_heads = RoIHeads(
            MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=7, sampling_ratio=2),
            TwoMLPHead(256 * 7 * 7, 1024), FastRCNNPredictor(1024, num_classes + 1),  # + 1: background
            fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None, score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)

    def forward(self, images, targets=None):
        """images [B, 1, H, W], targets: per-image {'boxes' XYXY, 'labels' 1..9} (training only) ->
        (logits [B, num_classes], per-image detections (eval mode), the four Faster R-CNN losses (training mode))."""
        set_task(self.backbone, 0)  # 0 = classification, 1 = detection; a no-op without MoE-LoRA
        tokens = self.backbone(images)
        pooled = tokens['3'].mean(dim=(-2, -1)) if self.spatial_backbone else tokens.mean(dim=1)
        logits = self.cls_head(pooled)
        if self.task_routing:
            set_task(self.backbone, 1)
            tokens = self.backbone(images)
        if self.spatial_backbone:
            features = self.neck(tokens)
        else:
            h, w = images.shape[-2] // 16, images.shape[-1] // 16
            features = self.neck(tokens.transpose(1, 2).unflatten(2, (h, w)))
        if self.detail_branch is not None:
            features = self.detail_branch(images, features)
        # Precision rule: the box coder casts anchors to the dtype of the regression output, so in bf16 every
        # coordinate near 512 would snap to a 2 px grid. RPN and RoIHeads therefore run in fp32.
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        return logits, detections, {**rpn_losses, **roi_losses}
