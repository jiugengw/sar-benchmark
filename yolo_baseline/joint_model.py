"""Joint YOLO detection and image-classification model used by Phase 1."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel

from yolo_baseline.image_head import ImageClassifierHead, pool_detection_features


def detection_feature_dim(detect_head: torch.nn.Module) -> int:
    towers = getattr(detect_head, "cv2", None)
    if not towers:
        raise ValueError("Expected a YOLO Detect head with per-scale cv2 towers")
    channels = []
    for tower in towers:
        first_conv = next((layer for layer in tower.modules() if isinstance(layer, torch.nn.Conv2d)), None)
        if first_conv is None:
            raise ValueError("Could not infer the input channels for a Detect feature level")
        channels.append(int(first_conv.in_channels))
    return sum(channels)


class JointDetectionModel(DetectionModel):
    """DetectionModel that adds image-label loss on its Detect input feature maps."""

    def __init__(
        self,
        cfg: str | dict[str, Any],
        nc: int = 80,
        ch: int = 3,
        verbose: bool = True,
        class_weights: list[float] | None = None,
        classification_loss_weight: float = 0.5,
        classifier_dropout: float = 0.2,
    ):
        super().__init__(cfg=cfg, nc=nc, ch=ch, verbose=verbose)
        feature_dim = detection_feature_dim(self.model[-1])
        self.image_classifier = ImageClassifierHead(feature_dim, num_classes=nc, dropout=classifier_dropout)
        weights = class_weights or [1.0] * nc
        if len(weights) != nc:
            raise ValueError(f"Expected {nc} image-class weights, found {len(weights)}")
        self.register_buffer("image_class_weights", torch.tensor(weights, dtype=torch.float32))
        self.classification_loss_weight = float(classification_loss_weight)
        self._joint_feature_maps = None
        self.model[-1].register_forward_pre_hook(self._capture_detect_inputs)

    def _capture_detect_inputs(self, _module, inputs) -> None:
        self._joint_feature_maps = inputs[0]

    def loss(self, batch: dict, preds=None):
        """Return the detection loss plus lambda-weighted image-classification loss."""
        if preds is None:
            self._joint_feature_maps = None
        detection_loss, detection_items = super().loss(batch, preds)
        feature_maps = self._joint_feature_maps
        if feature_maps is None:
            raise RuntimeError("The YOLO Detect feature hook did not capture this batch")

        features = pool_detection_features(feature_maps)
        logits = self.image_classifier(features)
        targets = batch.get("sar_image_class")
        if targets is None:
            # Ultralytics' detector validator calls loss(batch, preds) on its
            # own batches, which do not include the image-class labels attached
            # by JointDetectionTrainer.preprocess_batch. Classification metrics
            # are evaluated separately by the trainer's labeled validation
            # loader, so this path should report detector loss only.
            if preds is None:
                raise KeyError("Joint training batches must carry aligned sar_image_class targets")
            valid = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
            image_classification_loss = logits.sum() * 0.0
        else:
            valid = batch.get("sar_image_class_valid", targets >= 0)
        if targets is not None and valid.any():
            image_classification_loss = F.cross_entropy(
                logits[valid],
                targets[valid],
                weight=self.image_class_weights.to(dtype=logits.dtype),
            )
        elif targets is not None:
            image_classification_loss = logits.sum() * 0.0

        classification_term = self.classification_loss_weight * image_classification_loss * int(valid.sum())
        if detection_loss.ndim == 0:
            total = detection_loss + classification_term
        else:
            total = detection_loss.clone()
            total[0] = total[0] + classification_term
        if isinstance(detection_items, dict):
            named_detection = {str(name): value.detach() for name, value in detection_items.items()}
        else:
            values = detection_items.detach().reshape(-1)
            names = ("box_loss", "detector_cls_loss", "dfl_loss")
            named_detection = {
                names[index] if index < len(names) else f"detector_loss_{index}": value
                for index, value in enumerate(values)
            }
        named_detection["image_cls_loss"] = image_classification_loss.detach()
        return total, named_detection
