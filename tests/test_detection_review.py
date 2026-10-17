"""Diagnostic matching must distinguish duplicates, wrong labels and loose boxes."""
from tools.inspect_detections import diagnose


def test_matching_counts_and_ground_truth_identity():
    image = {'width': 512, 'height': 512}
    truth = [{'category_id': 1, 'bbox': [0, 0, 8, 8]}]
    predictions = [{'category_id': c, 'bbox': [0, 0, 8, 8], 'score': s}
                   for c, s in [(1, .9), (1, .8), (2, .7)]]
    review = diagnose(image, truth, predictions, .3, .5)
    assert review['counts']['matched'] == 1
    assert review['counts']['duplicate'] == 1
    assert review['counts']['wrong_class'] == 1
    assert review['counts']['matched_small'] == 1
    assert review['counts']['missed'] == 0
    assert review['predictions'][0]['matched_gt_index'] == 0
    assert review['predictions'][1]['matched_gt_index'] is None


def test_strict_overlap_and_confidence_change_misses():
    image = {'width': 512, 'height': 512}
    truth = [{'category_id': 1, 'bbox': [0, 0, 8, 8]}]
    predictions = [{'category_id': 1, 'bbox': [2, 0, 8, 8], 'score': .4}]
    assert diagnose(image, truth, predictions, .3, .5)['counts']['matched'] == 1
    strict = diagnose(image, truth, predictions, .3, .75)
    assert strict['counts']['localization'] == 1
    assert strict['counts']['missed_small'] == 1
    assert diagnose(image, truth, predictions, .5, .5)['counts']['missed'] == 1
