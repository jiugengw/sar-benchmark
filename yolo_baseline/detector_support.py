"""Small architecture adapters for Ultralytics detector wrappers."""

from __future__ import annotations

from pathlib import Path


def detector_class(architecture: str):
    """Return the requested Ultralytics detector wrapper."""
    from ultralytics import RTDETR, YOLO

    if architecture == "yolo":
        return YOLO
    if architecture == "rtdetr":
        return RTDETR
    if architecture == "deformable_detr":
        return None
    raise ValueError(f"Unsupported detector architecture: {architecture}")


def load_detector(checkpoint: str | Path, architecture: str = "yolo"):
    """Load a detector checkpoint using its architecture-specific wrapper."""
    if architecture == "deformable_detr":
        from deformable_detr_support import DeformableDetrAdapter

        return DeformableDetrAdapter(checkpoint)
    return detector_class(architecture)(str(checkpoint))


def detection_head(model):
    """Return the terminal module receiving multi-scale detector feature maps."""
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "backbone"):
        return inner_model.backbone
    modules = getattr(model, "model", None)
    if modules is None or not len(modules):
        raise ValueError("Detector does not expose a terminal multi-scale prediction module")
    return modules[-1]


def feature_maps_from_hook(architecture: str, inputs, output):
    """Extract feature maps from a forward hook on the chosen detector family."""
    if architecture == "deformable_detr":
        feature_pairs = output[0]
        return [feature for feature, _mask in feature_pairs]
    return inputs[0]
