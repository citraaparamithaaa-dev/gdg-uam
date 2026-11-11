"""NumPy-only inference: load numerical weights without JIT compilation."""
from pathlib import Path
import numpy as np
from fast_numpy_features import features


class Model:
    def __init__(self):
        with np.load(Path(__file__).with_name('weights.npz'), allow_pickle=False) as weights:
            self.indices = weights['indices']
            for key in ('roots', 'feature', 'threshold', 'left', 'right', 'value',
                        'missing_left', 'initial'):
                setattr(self, key, weights[key])

    def scores(self, vector):
        nodes = self.roots.copy()
        active = np.flatnonzero(self.feature[nodes] >= 0)
        while len(active):
            current = nodes[active]
            observed = vector[self.feature[current]]
            go_left = np.where(np.isnan(observed), self.missing_left[current],
                               observed <= self.threshold[current])
            nodes[active] = np.where(go_left, self.left[current], self.right[current])
            active = np.flatnonzero(self.feature[nodes] >= 0)
        increments = self.value[nodes].reshape(-1, len(self.initial))
        # Keep the sequential per-class accumulation order of the trained model.
        return np.cumsum(np.vstack([self.initial, increments]), axis=0)[-1]

    def predict(self, sample):
        vector = features(sample.series)[self.indices]
        return int(np.argmax(self.scores(vector)))
