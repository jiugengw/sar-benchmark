"""Deterministic class-stratified partitions and fold score summaries."""
import statistics
from collections import Counter


def stratified_folds(labels, folds=5, seed=0):
    from sklearn.model_selection import StratifiedKFold

    if folds < 2 or not labels or min(Counter(labels).values()) < folds:
        raise ValueError('Each class needs at least one image per fold; reduce folds or increase data.')
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [(train.tolist(), val.tolist()) for train, val in splitter.split(labels, labels)]


def summarize(scores):
    keys = ('accuracy', 'macro_f1', 'balanced_accuracy', 'mAP', 'AP50', 'delta_m')
    return {key: {'mean': statistics.mean(row[key] for row in scores),
                  'std': statistics.stdev(row[key] for row in scores) if len(scores) > 1 else 0.0}
            for key in keys}
