"""Fit an image into a 512 x 512 canvas without upscaling (competition rule: input at most 512 x 512, no upscaling).

Images larger than the canvas are shrunk to fit; smaller ones (a quarter of the data is 256 x 256) keep their native
size. Either way the image sits in the top-left corner and the rest is padded with zeros, so a box keeps its pixel
size and only the canvas grows. Used for training (as a Dome transform) and for prediction (fit_image).
"""
from __future__ import annotations


def fit_scale(width: int, height: int, size: int) -> float:
    """Factor applied to the image: 1 unless it is larger than the canvas."""
    return min(1.0, size / max(width, height))


def fit_image(image, size: int):
    """PIL image -> (PIL image on a size x size zero canvas, scale used)."""
    from PIL import Image

    scale = fit_scale(*image.size, size)
    if scale < 1.0:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                             Image.Resampling.BILINEAR)
    canvas = Image.new(image.mode, (size, size))
    canvas.paste(image, (0, 0))
    return canvas, scale


class FitPad:
    """Dome-DETR transform on (PIL image, target, dataset) samples; replaces Dome's Resize.

    target['orig_size'] becomes the canvas extent in original pixels, so Dome's own holdout evaluation (which scales
    normalised boxes by orig_size) still returns boxes in original pixels."""

    def __init__(self, size: int = 512):
        self.size = size

    def __call__(self, *inputs):
        import torch
        from src.data._misc import convert_to_tv_tensor

        sample = inputs if len(inputs) > 1 else inputs[0]
        image, target, *rest = sample
        canvas, scale = fit_image(image, self.size)
        if 'boxes' in target:
            boxes = target['boxes'].as_subclass(torch.Tensor) * scale
            target['boxes'] = convert_to_tv_tensor(boxes, key='boxes', box_format='xyxy',
                                                   spatial_size=(self.size, self.size))
        if 'orig_size' in target:
            extent = self.size / scale
            target['orig_size'] = torch.as_tensor([extent, extent], dtype=torch.float32)
        return (canvas, target, *rest)


def register_transforms() -> None:
    """Register FitPad and torchvision's vertical flip with Dome's config system (call after use_dome())."""
    import torchvision.transforms.v2 as T
    from src.core import register

    register()(T.RandomVerticalFlip)
    register()(FitPad)
