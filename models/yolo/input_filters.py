"""Deterministic input filters: what goes into the detector's 3 input channels.

By default every grayscale SAR image is copied into all 3 channels. A filter puts different, fixed views of the same
image into them instead, still one image and one forward pass:

    water      R = SAR image, G = despeckled image, B = water map (dark, smooth regions: rivers, lakes, sea)
    bilateral  all 3 = bilateral-filtered image (5 px, sigma 30 grey levels / 3 px): edge-preserving smoothing
    gauss1     all 3 = Gaussian-blurred image (sigma 1 px)

Holdout twins differ from their training copies mostly by smoothing; the same smoothing on every image brings a pair
much closer (residual after a gain/offset fit 0.27 -> 0.12 of the image's spread for bilateral, 0.12 for gauss1),
keeping 83% / 61% of the small-scale contrast (`twin_fixed_filter.py`).

The water map is computed from the image alone, with no labels: SAR water reflects the radar away, so it is dark and
smooth after despeckling. Bridges sit where roads cross it.

The same function filters training images as Ultralytics reads them (install_training_filter, no extra files: the
cluster allows 100,000 files), the submission code's images (make_submission.py) and the image classifier's images
(train_image_head.py). Select it with the environment variable SAR_INPUT_FILTER, so a run's training and prediction
cannot use different inputs.
"""
from __future__ import annotations

import os

import numpy as np

FILTERS = ('none', 'water', 'bilateral', 'gauss1')


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
    if name in ('bilateral', 'gauss1'):
        import cv2

        smooth = cv2.bilateralFilter(gray, 5, 30, 3) if name == 'bilateral' else cv2.GaussianBlur(gray, (0, 0), 1.0)
        return np.dstack([smooth, smooth, smooth])
    raise ValueError(f'unknown filter {name!r}; choose from {FILTERS}')


def install_training_filter() -> None:
    """Filter every image Ultralytics' training and validation datasets read, as it is read.

    Wraps the image reader of ultralytics.data.base, which BaseDataset.load_image uses. Our dataset images are
    grayscale PNGs that OpenCV reads as 3 identical channels; the wrapper turns each into the filtered image, in BGR
    order like OpenCV, so Ultralytics' usual BGR-to-RGB step gives the network [SAR, despeckled, water map], the
    same order the submission code writes. Does nothing when SAR_INPUT_FILTER is unset or 'none'."""
    name = active_filter()
    if name == 'none':
        return
    import ultralytics.data.base as base

    original = base.imread
    if getattr(original, 'sar_input_filter', None):
        return  # already installed

    def imread(filename, *args, **kwargs):
        image = original(filename, *args, **kwargs)
        if image is None:
            return image
        gray = image[..., 0] if image.ndim == 3 else image
        return np.ascontiguousarray(apply(name, np.ascontiguousarray(gray))[..., ::-1])

    imread.sar_input_filter = name
    base.imread = imread
    print(f'Input filter {name!r} applied to every training and validation image as it is read', flush=True)
