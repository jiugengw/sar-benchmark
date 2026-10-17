"""tools/build_holdout_viewer.py: per-image matching and AP50 used by the holdout viewer."""
import pytest

from tools.build_holdout_viewer import image_ap50, match_image


def test_match_counts_hits_false_positives_and_misses():
    gt = [(1, [0, 0, 10, 10]), (1, [50, 50, 10, 10])]
    dets = [(1, [0, 0, 10, 10], 0.9),      # hit
            (1, [1, 1, 10, 10], 0.8),      # duplicate of a matched box: false positive
            (2, [50, 50, 10, 10], 0.7),    # wrong class: false positive
            (1, [50, 50, 10, 10], 0.1)]    # below the score threshold: ignored
    m = match_image(gt, dets, score_thr=0.3)
    assert (m['tp'], m['fp'], m['fn']) == (1, 2, 1)
    assert m['det_match'] == [0, -1, -1, None]


def test_image_ap50_is_one_for_perfect_detections_and_none_without_ground_truth():
    gt = [(3, [0, 0, 8, 8])]
    assert image_ap50(gt, [(3, [0, 0, 8, 8], 0.9)]) == pytest.approx(1.0)
    assert image_ap50([], [(3, [0, 0, 8, 8], 0.9)]) is None
