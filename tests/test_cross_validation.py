import pytest

from sarbench.cross_validation import stratified_folds, summarize


def test_folds_are_disjoint_stratified_and_cover_every_image_once():
    labels = [label for label in range(9) for _ in range(13)]
    folds = stratified_folds(labels, 5, 7)
    assert folds == stratified_folds(labels, 5, 7)
    assert folds != stratified_folds(labels, 5, 8)
    held_out = []
    for train, val in folds:
        assert not set(train) & set(val)
        assert set(train) | set(val) == set(range(len(labels)))
        assert all(sum(labels[i] == label for i in val) in (2, 3) for label in range(9))
        held_out.extend(val)
    assert sorted(held_out) == list(range(len(labels)))


def test_rejects_classes_too_small_for_folds():
    with pytest.raises(ValueError, match='Each class'):
        stratified_folds([0] * 10 + [1] * 4)


def test_summary_uses_sample_standard_deviation():
    keys = ('accuracy', 'macro_f1', 'balanced_accuracy', 'mAP', 'AP50', 'delta_m')
    summary = summarize([dict.fromkeys(keys, 1.0), dict.fromkeys(keys, 3.0)])
    assert summary['mAP']['mean'] == 2.0
    assert summary['mAP']['std'] == pytest.approx(2 ** 0.5)
