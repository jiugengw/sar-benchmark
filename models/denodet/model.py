"""Official DenoDet V2 GFL with classification on the same forward pass."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmengine.evaluator import BaseMetric
from mmengine.hooks import Hook
from mmdet.models.detectors import GFL
from mmdet.registry import HOOKS, METRICS, MODELS


@MODELS.register_module()
class MultiTaskDenoDet(GFL):
    def __init__(self, num_image_classes=9, classification_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.classification_weight = classification_weight
        channels = self.neck.out_channels * self.neck.num_outs
        self.image_head = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 256),
                                        nn.GELU(), nn.Dropout(0.2), nn.Linear(256, num_image_classes))

    def image_logits(self, features):
        return self.image_head(torch.cat([feature.mean((2, 3)) for feature in features], dim=1))

    def loss(self, batch_inputs, batch_data_samples):
        features = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(features, batch_data_samples)
        targets = []
        for sample in batch_data_samples:
            labels = sample.gt_instances.labels
            if len(labels) == 0 or not torch.all(labels == labels[0]):
                raise ValueError("Course images must have at least one box and one image class")
            targets.append(labels[0])
        losses["loss_image_cls"] = self.classification_weight * F.cross_entropy(
            self.image_logits(features), torch.stack(targets))
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        features = self.extract_feat(batch_inputs)
        predictions = self.bbox_head.predict(features, batch_data_samples, rescale=rescale)
        logits = self.image_logits(features)
        samples = self.add_pred_to_datasample(batch_data_samples, predictions)
        for sample, scores in zip(samples, logits):
            sample.image_logits = scores
        return samples


@METRICS.register_module()
class ImageClassificationMetric(BaseMetric):
    default_prefix = "image"

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            labels = sample["gt_instances"]["labels"]
            self.results.append((int(labels[0]), int(sample["image_logits"].argmax())))

    def compute_metrics(self, results):
        matrix = np.zeros((9, 9), dtype=np.int64)
        for truth, prediction in results:
            matrix[truth, prediction] += 1
        tp = np.diag(matrix)
        f1 = np.divide(2 * tp, matrix.sum(0) + matrix.sum(1),
                       out=np.zeros(9), where=(matrix.sum(0) + matrix.sum(1)) != 0)
        return dict(macro_f1=float(f1.mean()), accuracy=float(tp.sum() / matrix.sum()))


@HOOKS.register_module()
class FiniteLossHook(Hook):
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        for name, value in outputs.items():
            if "loss" in name and not np.isfinite(float(value)):
                raise FloatingPointError(f"Nonfinite loss at epoch {runner.epoch + 1}, batch {batch_idx}: {name}={value}")
