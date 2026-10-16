"""Hugging Face Deformable DETR helpers and prediction adapter."""

from __future__ import annotations

import time
import re
from pathlib import Path
from types import SimpleNamespace

import torch


def make_p2_model(base_checkpoint: str | Path, class_names: list[str]):
    """Build a four-level P2 Deformable DETR and transfer pretrained P3-P5 weights."""
    from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection

    base = Path(base_checkpoint)
    source_config = DeformableDetrConfig.from_pretrained(base, local_files_only=True)
    source_config.use_pretrained_backbone = False
    source = DeformableDetrForObjectDetection.from_pretrained(
        base, config=source_config, local_files_only=True
    )
    p2_config = DeformableDetrConfig.from_dict(source_config.to_dict())
    p2_config.use_pretrained_backbone = False
    p2_config.backbone_kwargs = {**(p2_config.backbone_kwargs or {}), "out_indices": [1, 2, 3, 4]}
    p2_config.num_feature_levels = 4
    p2_config.num_labels = len(class_names)
    p2_config.id2label = {index: name for index, name in enumerate(class_names)}
    p2_config.label2id = {name: index for index, name in enumerate(class_names)}
    target = DeformableDetrForObjectDetection(p2_config)
    source_state = source.state_dict()
    target_state = target.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    backbone_transformer = projections = 0
    for key, value in source_state.items():
        match = re.match(r"model\.input_proj\.(\d+)\.(.*)", key)
        if match:
            source_level = int(match.group(1))
            # The source's first three projections consume P3/P4/P5. P2 is a new level.
            if source_level >= 3:
                continue
            target_key = f"model.input_proj.{source_level + 1}.{match.group(2)}"
            if target_key in target_state and target_state[target_key].shape == value.shape:
                transferred[target_key] = value
                projections += 1
        elif key in target_state and target_state[key].shape == value.shape:
            transferred[key] = value
            backbone_transformer += 1
    missing, unexpected = target.load_state_dict(transferred, strict=False)
    if unexpected or backbone_transformer < 200 or projections < 10:
        raise RuntimeError(
            f"Insufficient Deformable DETR transfer: backbone/transformer={backbone_transformer}, "
            f"projections={projections}, unexpected={unexpected[:8]}"
        )
    transfer = {
        "backbone_transformer_tensors": backbone_transformer,
        "p3_p5_projection_tensors": projections,
        "new_p2_projection_parameters": sum(parameter.numel() for parameter in target.model.input_proj[0].parameters()),
        "skipped_target_keys": len(missing),
    }
    del source
    return target, p2_config, transfer


def detection_feature_maps_from_backbone_output(output):
    """Return unprojected multi-scale tensors from Deformable DETR's backbone hook."""
    pairs = output[0]
    return [feature for feature, _mask in pairs]


class PredictionBoxes:
    def __init__(self, xyxy: torch.Tensor, classes: torch.Tensor, scores: torch.Tensor):
        self.xyxy = xyxy
        self.cls = classes
        self.conf = scores

    def __len__(self):
        return int(self.xyxy.shape[0])


class DeformableDetrAdapter:
    """Expose a Deformable DETR model through the predictor interface used in this repo."""

    def __init__(self, checkpoint: str | Path, device: torch.device | None = None):
        from transformers import AutoImageProcessor, DeformableDetrConfig, DeformableDetrForObjectDetection

        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(self.checkpoint)
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = DeformableDetrConfig.from_pretrained(self.checkpoint, local_files_only=True)
        config.use_pretrained_backbone = False
        self.model = DeformableDetrForObjectDetection.from_pretrained(
            self.checkpoint, config=config, local_files_only=True
        ).to(self.device).eval()
        self.image_processor = AutoImageProcessor.from_pretrained(self.checkpoint, local_files_only=True)
        self.image_processor.size = {"height": 512, "width": 512}
        self.names = {int(key): str(value) for key, value in self.model.config.id2label.items()}
        self.overrides = {"imgsz": 512, "task": "detect"}

    def prepare_batch(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Normalize already-resized RGB tensors in the same way as the image processor."""
        mean = torch.tensor(self.image_processor.image_mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.image_processor.image_std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        return {
            "pixel_values": (images - mean) / std,
            "pixel_mask": torch.ones((images.shape[0], images.shape[2], images.shape[3]),
                                     dtype=torch.long, device=images.device),
        }

    def predict(self, source, imgsz=512, conf=0.001, max_det=300, batch=1, stream=False, **_kwargs):
        from PIL import Image
        import numpy as np

        if isinstance(source, (str, Path)):
            source_path = Path(source)
            paths = sorted(
                (path for path in source_path.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}),
                key=lambda path: path.name,
            ) if source_path.is_dir() else [source_path]
        else:
            paths = [Path(path) for path in source]
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError("A Deformable DETR prediction input image is missing")

        def results():
            for start in range(0, len(paths), max(1, int(batch))):
                batch_paths = paths[start:start + max(1, int(batch))]
                preprocessing_start = time.perf_counter()
                images = []
                for path in batch_paths:
                    with Image.open(path) as image:
                        resized = image.convert("RGB").resize((int(imgsz), int(imgsz)), Image.Resampling.BILINEAR)
                        images.append(np.asarray(resized, dtype=np.uint8).copy())
                processed = self.image_processor(images=images, return_tensors="pt")
                pixel_values = processed["pixel_values"].to(self.device)
                pixel_mask = processed.get("pixel_mask")
                if pixel_mask is None:
                    pixel_mask = torch.ones(pixel_values.shape[:1] + pixel_values.shape[-2:], dtype=torch.long)
                pixel_mask = pixel_mask.to(self.device)
                preprocess_ms = (time.perf_counter() - preprocessing_start) * 1000 / len(batch_paths)
                inference_start = time.perf_counter()
                with torch.inference_mode():
                    outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
                inference_ms = (time.perf_counter() - inference_start) * 1000 / len(batch_paths)
                postprocess_start = time.perf_counter()
                with torch.inference_mode():
                    sizes = torch.tensor([[imgsz, imgsz]] * len(batch_paths), device=self.device)
                    predictions = self.image_processor.post_process_object_detection(
                        outputs, target_sizes=sizes, threshold=float(conf)
                    )
                for path, prediction in zip(batch_paths, predictions):
                    scores = prediction["scores"]
                    if len(scores) > max_det:
                        indices = scores.topk(int(max_det)).indices
                        scores = scores[indices]
                        classes = prediction["labels"][indices]
                        boxes = prediction["boxes"][indices]
                    else:
                        classes = prediction["labels"]
                        boxes = prediction["boxes"]
                    yield SimpleNamespace(
                        path=str(path),
                        boxes=PredictionBoxes(boxes, classes, scores),
                        speed={
                            "preprocess": preprocess_ms,
                            "inference": inference_ms,
                            "postprocess": (time.perf_counter() - postprocess_start) * 1000 / len(batch_paths),
                        },
                    )

        iterator = results()
        return iterator if stream else list(iterator)
