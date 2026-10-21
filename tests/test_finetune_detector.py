"""models/yolo/finetune_detector.py: the NWD box similarity and the speckle augmentation."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from models.yolo.finetune_detector import SpeckleNoise, normalized_wasserstein  # noqa: E402


def test_nwd_is_one_for_identical_boxes_and_smooth_without_overlap():
    box = torch.tensor([[0.0, 0.0, 8.0, 8.0]])
    assert normalized_wasserstein(box, box, 16).item() == pytest.approx(1.0, abs=1e-4)
    shifted = normalized_wasserstein(box, box + torch.tensor([1.0, 0, 1.0, 0]), 16).item()
    apart = normalized_wasserstein(box, box + 20, 16).item()
    assert 0.9 < shifted < 1.0  # a 1 px shift on an 8 px box costs little (IoU would drop to ~0.78)
    assert 0.0 < apart < shifted  # still a signal when the boxes do not overlap at all


def test_speckle_keeps_brightness_and_is_identical_across_channels():
    np.random.seed(0)
    image = np.full((64, 64, 3), 100, np.uint8)
    out = SpeckleNoise(p=1.0, looks=4)({'img': image.copy()})['img']
    assert abs(out.mean() - 100) < 5 and 40 < out.std() < 60
    assert (out[..., 0] == out[..., 2]).all()
    assert (SpeckleNoise(p=0.0)({'img': image.copy()})['img'] == image).all()


def test_repeat_labels_duplicates_only_images_with_chosen_classes():
    from models.yolo.finetune_detector import repeat_labels
    labels = [{'im_file': 'a', 'cls': np.array([[2.0]])}, {'im_file': 'b', 'cls': np.array([[0.0], [0.0]])},
              {'im_file': 'c', 'cls': np.zeros((0, 1))}]
    out = repeat_labels(labels, {2: 3, 6: 3})
    assert [label['im_file'] for label in out] == ['a', 'a', 'a', 'b', 'c']


def test_repeat_labels_uses_the_largest_factor_of_an_images_classes():
    from models.yolo.finetune_detector import repeat_labels
    labels = [{'im_file': 'a', 'cls': np.array([[1.0], [2.0]])}, {'im_file': 'b', 'cls': np.array([[1.0]])}]
    out = repeat_labels(labels, {1: 2, 2: 4})
    assert [label['im_file'] for label in out] == ['a'] * 4 + ['b'] * 2
