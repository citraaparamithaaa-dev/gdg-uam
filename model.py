"""Portable single-sample inference for an exported gradient-boosted classifier."""
from pathlib import Path
import numpy as np
from numba import njit


@njit(cache=False)
def raw_scores(x, roots, feature, threshold, left, right, value, missing_left, initial):
    scores = initial.copy()
    for tree in range(len(roots)):
        node = roots[tree]
        while feature[node] >= 0:
            v = x[feature[node]]
            if np.isnan(v):
                node = left[node] if missing_left[node] else right[node]
            else:
                node = left[node] if v <= threshold[node] else right[node]
        scores[tree % len(scores)] += value[node]
    return scores


class Model:
    def __init__(self):
        with np.load(Path(__file__).with_name("weights.npz"), allow_pickle=False) as packed:
            self.indices = packed["indices"]
            self.kind = str(packed["kind"].item())
            self.trees = tuple(packed[key] for key in (
                "roots", "feature", "threshold", "left", "right", "value", "missing_left", "initial"))
        if self.kind == "core":
            from core_features import features
            self.transform = lambda sample: features(sample.series)
        elif self.kind == "structure":
            from structure_features import features
            self.transform = lambda sample: features(sample.series)
        elif self.kind == "signal":
            from signal_features import features
            self.transform = lambda sample: features(sample.series)
        else:
            from baseline_features import features
            self.transform = features
        # Compile CPU routines during setup, before the scored prediction phase.
        from competition_api import Sample
        self.predict(Sample(np.zeros((4, 256), dtype=np.float32),
                            {"sample_rate": 1.0, "group": "A", "level": 0.0}))

    def predict(self, x):
        vector = self.transform(x)[self.indices]
        return int(np.argmax(raw_scores(vector, *self.trees)))
