"""Pairwise matcher: gradient-boosted trees over the features in features.py.

Trained on every blocked pair of 100k fit-split S1 entities (9.7M pairs, 3.3% positive,
no negative subsampling, so scores stay close to calibrated for decide.py). Final match
selection is per entity (decide.select_expected_f), not by this class's threshold.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


DEFAULT_THRESHOLD = 0.60


class PairMatcher:
    def __init__(self, threshold=DEFAULT_THRESHOLD, **kw):
        self.threshold = threshold
        params = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
                      l2_regularization=1.0, random_state=0)
        params.update(kw)
        self.clf = HistGradientBoostingClassifier(**params)

    def fit(self, X, y):
        self.clf.fit(np.asarray(X, dtype=np.float32), np.asarray(y))
        return self

    def score(self, X):
        return self.clf.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]


def macro_f05(predictions, truth):
    """Official metric: F0.5 per Source-1 entity, averaged. Singletons included —
    an empty prediction for an entity with no true matches scores 1.0."""
    total = 0.0
    for key, true_set in truth.items():
        pred = set(predictions.get(key, ()))
        if not pred and not true_set:
            total += 1.0
            continue
        if not pred or not true_set:
            continue
        tp = len(pred & true_set)
        if not tp:
            continue
        precision = tp / len(pred)
        recall = tp / len(true_set)
        total += (1.25 * precision * recall) / (0.25 * precision + recall)
    return total / len(truth) if truth else 0.0
