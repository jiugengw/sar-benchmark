import pytest
import torch

from sarbench.inference import calibrate_detections


def test_calibration_uses_one_based_detection_categories_and_preserves_boxes():
    boxes = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    detections = [{'boxes': boxes, 'labels': torch.tensor([1, 3]), 'scores': torch.tensor([0.8, 0.6])}]
    logits = torch.tensor([[0., 0., 1.]])
    result = calibrate_detections(detections, logits, 1.)
    assert result[0]['boxes'] is boxes
    assert torch.allclose(result[0]['scores'], detections[0]['scores'] * logits.softmax(-1)[0, [0, 2]])
    assert torch.equal(detections[0]['scores'], torch.tensor([0.8, 0.6]))
    assert calibrate_detections(detections, logits, 0.) is detections


@pytest.mark.parametrize('power', [-1., float('nan'), float('inf')])
def test_invalid_calibration_power(power):
    with pytest.raises(ValueError):
        calibrate_detections([], torch.empty(0, 9), power)
