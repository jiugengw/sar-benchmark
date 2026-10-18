"""Deterministic input filters: what goes into the detector's 3 input channels.

By default every grayscale SAR image is copied into all 3 channels. A filter puts different, fixed views of the same
image into them instead, still one image and one forward pass:

    water    R = SAR image, G = despeckled image, B = water map (dark, smooth regions: rivers, lakes, sea)

The water map is computed from the image alone, with no labels: SAR water reflects the radar away, so it is dark and
smooth after despeckling. Bridges sit where roads cross it.

The same function is used to build the training images (build_variant_dataset.py), by the submission code
(make_submission.py) and by the image classifier (train_image_head.py). Select it with the environment variable
SAR_INPUT_FILTER, so a run's training and prediction cannot use different inputs.
"""
from __future__ import annotations

import os

import numpy as np

FILTERS = ('none', 'water')


def active_filter() -> str:
    name = os.environ.get('SAR_INPUT_FILTER', 'none').strip() or 'none'
    if name not in FILTERS:
        raise ValueError(f'SAR_INPUT_FILTER={name!r}; choose from {FILTERS}')
    return name


def water(gray: np.ndarray) -> np.ndarray:
    """uint8 HxW grayscale -> uint8 HxWx3 [SAR, despeckled, water map]."""
    import cv2

    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError(f'expected a uint8 grayscale image, got {gray.dtype} {gray.shape}')
    smooth = cv2.medianBlur(gray, 5)
    smooth = cv2.GaussianBlur(smooth, (0, 0), 2.0)
    lo, mid = np.percentile(smooth, [1, 30])
    # 1 for the darkest smooth regions, fading to 0 at the image's 30th brightness percentile
    wet = np.clip((mid - smooth.astype(np.float32)) / max(float(mid - lo), 1.0), 0.0, 1.0)
    return np.dstack([gray, smooth, np.round(wet * 255).astype(np.uint8)])


def apply(name: str, gray: np.ndarray) -> np.ndarray:
    """Apply filter `name` to a uint8 grayscale image and return the uint8 HxWx3 network input (RGB order)."""
    if name == 'none':
        return np.dstack([gray, gray, gray])
    if name == 'water':
        return water(gray)
    raise ValueError(f'unknown filter {name!r}; choose from {FILTERS}')
