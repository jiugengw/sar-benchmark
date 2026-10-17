import numpy as np

from yolo_baseline.quick_box_bias import apply, fit


def test_fit_corrects_edge_direction_with_fixed_shrinkage():
    images = {1: {'width': 512, 'height': 512}}
    predictions = [dict(image_id=1, category_id=1, score=0.9,
                        bbox=[20*i+10, 10, 10, 10]) for i in range(12)]
    annotations = [dict(image_id=1, category_id=1,
                        bbox=[20*i+11, 12, 10, 10]) for i in range(12)]
    params = fit({1}, predictions, annotations, images)
    assert params['1:512']['matches'] == 12
    np.testing.assert_allclose(params['1:512']['edge_offsets_512'], [0.5, 1, 0.5, 1])
    corrected = apply(predictions, params, images)
    np.testing.assert_allclose(corrected[0]['bbox'], [10.5, 11, 10, 10])
    assert predictions[0]['bbox'] == [10, 10, 10, 10]
    assert corrected[0]['score'] == predictions[0]['score']


def test_apply_rescales_offsets_and_clips_to_image():
    images = {1: {'width': 256, 'height': 256}}
    predictions = [dict(image_id=1, category_id=2, score=0.7, bbox=[0, 0, 256, 256])]
    params = {'2:256': {'matches': 12, 'edge_offsets_512': [-1, -1, -1, -1]}}
    corrected = apply(predictions, params, images)
    np.testing.assert_allclose(corrected[0]['bbox'], [0, 0, 255.5, 255.5])
    assert corrected[0]['category_id'] == 2
    assert apply(predictions, {}, images) == predictions
