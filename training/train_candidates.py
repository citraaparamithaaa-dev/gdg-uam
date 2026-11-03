"""Controlled feature experiments using a single fixed training-only split.

The demo labels never enter fit. Results preserve separate development and demo metrics.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "4"
import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits
from competition_api import load_dataset
from signal_features import features, basic_shape_indices

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "examples"))
from baseline_model import features as baseline_features


def cached_features(ds, name, kind):
    cache = ROOT / "data" / f"features_{kind}_{name}.npy"
    if cache.exists():
        return np.load(cache)
    transform = (lambda x: features(x.series)) if kind == "signal" else baseline_features
    rows = []
    start = time.perf_counter()
    for i in range(len(ds)):
        rows.append(transform(ds[i]))
        if (i + 1) % 10000 == 0:
            print(f"{name} {kind}: {i+1}/{len(ds)}, {time.perf_counter()-start:.1f}s", flush=True)
    X = np.stack(rows)
    np.save(cache, X)
    print(f"{name} {kind} features {X.shape}: {time.perf_counter()-start:.1f}s", flush=True)
    return X


def metrics(y, pred):
    cm = confusion_matrix(y, pred, labels=np.arange(8))
    recall = cm.diagonal() / np.maximum(1, cm.sum(axis=1))
    term = np.maximum(.01, np.sqrt(cm.sum(axis=1)) * (1.33 * recall - .33) * recall**2)
    return {"accuracy": float(accuracy_score(y, pred)), "recall": recall.tolist(),
            "confusion_matrix": cm.tolist(), "finite_file_score": float(np.exp(np.log(term).mean()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", help="refit a selected candidate using all training rows")
    args = parser.parse_args()
    (ROOT / "experiments").mkdir(exist_ok=True)
    ds = load_dataset(ROOT / "data/train_100k.npz")
    split = np.load(ROOT / "data/team_split_ids.npz")
    tr, val = split["train"], split["val"]
    signal = cached_features(ds, "train", "signal")
    assert signal.shape[1] == 672, signal.shape
    # The compact model excludes directional lags and difference correlations.
    shape = basic_shape_indices()
    experiments = [
        ("shape_150", "signal", shape, 150, 15),
        ("relations_150", "signal", np.arange(signal.shape[1]), 150, 15),
        ("relations_300", "signal", np.arange(signal.shape[1]), 300, 15),
    ]
    if args.final:
        previous = json.loads((ROOT / f"experiments/{args.final}/metrics.json").read_text())
        kind = previous["feature_kind"]
        ids = np.array(previous["feature_indices"])
        if kind == "structure":
            X = np.load(ROOT / "data/features_structure_train.npy")
        else:
            X = signal if kind == "signal" else cached_features(ds, "train", "baseline")
        clf = HistGradientBoostingClassifier(**previous["settings"]).fit(X[:, ids], ds.labels)
        with (ROOT / f"experiments/{args.final}/final.pkl").open("wb") as f:
            pickle.dump({"classifier": clf, "indices": ids, "kind": kind,
                         "bias": np.asarray(previous.get("class_bias",np.zeros(8)))}, f)
        print(f"Final refit complete: {args.final}", flush=True)
        return
    baseline = cached_features(ds, "train", "baseline")
    experiments.insert(0, ("baseline_300", "baseline", np.arange(baseline.shape[1]), 300, 31))
    # Metadata and absolute level/scale ablation on otherwise identical baseline features.
    # baseline has 4 * 23 per-channel entries, then 6 pair correlations,
    # 2 numeric metadata, and 1 missing fraction: 101 entries in total.
    assert baseline.shape[1] == 101, baseline.shape
    shape_baseline = np.array([i for i in range(92) if i % 23 >= 2] + list(range(92, 98)))
    experiments.insert(1, ("baseline_shape_300", "baseline", shape_baseline, 300, 31))
    demo = load_dataset(ROOT / "data/test_demo.npz")
    demo_X = {kind: cached_features(demo, "demo", kind) for kind in ("baseline", "signal")}
    summaries = []
    for name, kind, ids, iterations, leaves in experiments:
        folder = ROOT / "experiments" / name
        folder.mkdir(exist_ok=True)
        if (folder / "metrics.json").exists():
            summaries.append(json.loads((folder / "metrics.json").read_text()))
            continue
        X = signal if kind == "signal" else baseline
        settings = dict(max_iter=iterations, max_leaf_nodes=leaves, learning_rate=.08,
                        l2_regularization=1., random_state=42, early_stopping=False)
        if kind == "baseline":
            # Exactly the settings from the organizer's make_classifier(seed=0).
            settings = dict(max_iter=300, learning_rate=.08, random_state=0)
        print(f"Training {name}: {len(tr)} rows, {len(ids)} features", flush=True)
        start = time.perf_counter()
        clf = HistGradientBoostingClassifier(**settings).fit(X[tr][:, ids], ds.labels[tr])
        elapsed = time.perf_counter() - start
        record = {"name": name, "feature_kind": kind, "feature_indices": ids.tolist(),
                  "settings": settings, "training_seconds": elapsed,
                  "validation": metrics(ds.labels[val], clf.predict(X[val][:, ids])),
                  "demo": metrics(demo.labels, clf.predict(demo_X[kind][:, ids])),
                  "training_rows": len(tr), "validation_rows": len(val), "split_seed": 42,
                  "feature_source_sha256": hashlib.sha256((ROOT / "signal_features.py").read_bytes()).hexdigest()}
        with (folder / "development.pkl").open("wb") as f:
            pickle.dump({"classifier": clf, "indices": ids, "kind": kind}, f)
        (folder / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"{name}: validation={record['validation']['accuracy']:.4f}, demo={record['demo']['accuracy']:.4f}, train={elapsed:.1f}s", flush=True)
        print("  validation recall:", np.round(record["validation"]["recall"], 3), flush=True)
        print("  demo recall:", np.round(record["demo"]["recall"], 3), flush=True)
        summaries.append(record)
    with (ROOT / "analysis/experiments.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "validation_accuracy", "demo_accuracy", "validation_min_recall", "demo_min_recall", "training_seconds"])
        for row in summaries:
            writer.writerow([row["name"], row["validation"]["accuracy"], row["demo"]["accuracy"],
                             min(row["validation"]["recall"]), min(row["demo"]["recall"]), row["training_seconds"]])


if __name__ == "__main__":
    main()
