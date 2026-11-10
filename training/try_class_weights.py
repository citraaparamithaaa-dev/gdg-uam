"""Reproduce the class-weighting experiment (rejected candidate).

Fits the core_structure_200 classifier with and without sample weights x2 on
classes 6/7, calibrates class offsets on the 5k calibration rows of each
split, and reports the untouched 15k check score for the seed-42 and seed-123
splits. The weighted fit improves both in-distribution checks but was
rejected because the shipped package must also survive the shifted demo set,
where the weighted model scored 2.23 against 2.565 for the unweighted one
(evidence/class_pairs.json). Demo labels are never used here.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "4"
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
SETTINGS = dict(max_iter=200, max_leaf_nodes=15, learning_rate=.08,
                l2_regularization=1., random_state=42, early_stopping=False,
                max_bins=128)


def score(y, pred):
    n = np.bincount(y, minlength=8)
    c = np.bincount(y[pred == y], minlength=8)
    rec = c / np.maximum(n, 1)
    terms = np.maximum(.01, np.sqrt(n) * (1.33 * rec - .33) * rec ** 2)
    return float(np.exp(np.log(terms).mean()))


def fit_bias(raw, labels):
    bias = np.zeros(8)
    best = score(labels, np.argmax(raw + bias, axis=1))
    for step in (1., .5, .25, .125, .0625):
        for _ in range(3):
            changed = False
            for k in range(8):
                origin = bias[k]
                chosen = origin
                for shift in (-step, step):
                    trial = bias.copy()
                    trial[k] = origin + shift
                    value = score(labels, np.argmax(raw + trial, axis=1))
                    if value > best + 1e-10:
                        best = value
                        chosen = trial[k]
                if chosen != origin:
                    bias[k] = chosen
                    changed = True
            bias -= bias.mean()
            if not changed:
                break
    return bias


def main(data="data/train_100k.npz"):
    from competition_api import load_dataset
    ds = load_dataset(HERE / data)
    y = np.asarray(ds.labels)
    from core_features import features
    indices = np.load(HERE.parent / "weights.npz", allow_pickle=False)["indices"]
    print("extracting features (cached afterwards) ...", flush=True)
    cache = HERE / f"data/features_core_{Path(data).stem}.npy"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        X = np.load(cache)
    else:
        X = np.stack([features(ds[i].series)[indices] for i in range(len(ds))])
        np.save(cache, X)

    for seed in (42, 123):
        train, val = train_test_split(np.arange(len(ds)), test_size=.2,
                                      random_state=seed, stratify=y)
        cal, check = train_test_split(val, train_size=5000, random_state=314,
                                      stratify=y[val])
        for name, weights in (("plain", None), ("weighted 6/7", [1, 1, 1, 1, 1, 1, 2, 2])):
            t0 = time.perf_counter()
            clf = HistGradientBoostingClassifier(**SETTINGS).fit(
                X[train], y[train],
                sample_weight=None if weights is None else np.asarray(weights)[y[train]])
            b = fit_bias(clf._raw_predict(X[cal]), y[cal])
            pred = np.argmax(clf._raw_predict(X[check]) + b, axis=1)
            print(f"seed {seed} {name:13s}: check {score(y[check], pred):.4f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
