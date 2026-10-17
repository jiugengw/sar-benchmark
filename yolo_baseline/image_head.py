"""Small supervised image-classification head shared by the YOLO experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ImageClassifierHead(nn.Module):
    """Classify a whole image from pooled features used by YOLO's detection head."""

    def __init__(self, feature_dim: int, num_classes: int = 9, dropout: float = 0.2,
                 hidden_dim: int = 0):
        super().__init__()
        layers = [nn.LayerNorm(feature_dim), nn.Dropout(dropout)]
        if hidden_dim:
            layers += [nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim or feature_dim, num_classes)]
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def pool_detection_features(feature_maps, pooling: str = "mean") -> torch.Tensor:
    """Pool every multi-scale detection feature map and concatenate channels."""
    if not isinstance(feature_maps, (tuple, list)) or not feature_maps:
        raise ValueError("Detection head did not receive a feature-map list")
    if pooling == "mean":
        return torch.cat([feature.float().mean(dim=(2, 3)) for feature in feature_maps], dim=1)
    if pooling == "spatial":
        # Keep coarse scene layout and sparse activations that global averaging can wash out.
        pooled = []
        for feature in feature_maps:
            feature = feature.float()
            pooled += [feature.mean(dim=(2, 3)), feature.amax(dim=(2, 3)),
                       torch.nn.functional.adaptive_avg_pool2d(feature, 2).flatten(1)]
        return torch.cat(pooled, dim=1)
    raise ValueError(f"Unknown classifier feature pooling: {pooling}")


def classification_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict:
    """Return accuracy, macro-F1, balanced accuracy and class-wise scores."""
    count = len(class_names)
    matrix = [[0 for _ in range(count)] for _ in range(count)]
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth)][int(prediction)] += 1
    per_class = {}
    recalls = []
    f1s = []
    for index, name in enumerate(class_names):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(matrix[row][index] for row in range(count))
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        recalls.append(recall)
        f1s.append(f1)
    correct = sum(matrix[index][index] for index in range(count))
    return {
        "accuracy": correct / max(1, len(y_true)),
        "macro_f1": sum(f1s) / max(1, count),
        "balanced_accuracy": sum(recalls) / max(1, count),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_columns": class_names,
    }


def load_classifier_head(path: Path, device: torch.device):
    """Load a saved head and return (module, checkpoint metadata)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    head = ImageClassifierHead(
        feature_dim=int(payload["feature_dim"]),
        num_classes=len(payload["class_names"]),
        dropout=float(payload.get("dropout", 0.2)),
        hidden_dim=int(payload.get("hidden_dim", 0)),
    )
    head.load_state_dict(payload["state_dict"], strict=True)
    return head.to(device).eval(), payload
