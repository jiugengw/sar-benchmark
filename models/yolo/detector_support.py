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


def alias_legacy_modules() -> None:
    """Let checkpoints pickled before the 2026-10-18 move (classes under ``yolo_baseline.*``) find ``models.yolo``."""
    import importlib
    import sys

    if "yolo_baseline" in sys.modules:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
    try:
        sys.modules["yolo_baseline"] = importlib.import_module("models.yolo")
        for name in ("image_head", "joint_model", "joint_rtdetr", "joint_trainer"):
            sys.modules[f"yolo_baseline.{name}"] = importlib.import_module(f"models.yolo.{name}")
    except ImportError:
        pass


def load_detector(checkpoint: str | Path, architecture: str = "yolo"):
    """Load a detector checkpoint using its architecture-specific wrapper."""
    alias_legacy_modules()
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import spd  # noqa: F401  checkpoints with an SPD first layer unpickle spd.SPDConv
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
