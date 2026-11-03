"""Predeclared structural-feature comparison on the same fixed split."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from competition_api import load_dataset
from train_candidates import metrics
from structure_features import extra_features, extra_features_batch, BASE_INDICES

ROOT = Path(__file__).resolve().parent


def matrix(ds, name):
    path = ROOT / f"data/features_structure_{name}.npy"
    if path.exists():
        return np.load(path)
    base = np.load(ROOT / f"data/features_signal_{name}.npy")[:,BASE_INDICES]
    extras = []
    start = time.perf_counter()
    for i in range(0,len(ds),512):
        extras.append(extra_features_batch(ds.series[i:i+512]))
        if i % 10240 == 0:
            print(f"Structure features {name}: {min(i+512,len(ds))}/{len(ds)} ({time.perf_counter()-start:.0f}s)",flush=True)
    X = np.column_stack([base, np.concatenate(extras)])
    np.save(path,X)
    return X


def main():
    ds = load_dataset(ROOT / "data/train_100k.npz")
    demo = load_dataset(ROOT / "data/test_demo.npz")
    X = matrix(ds,"train")
    D = matrix(demo,"demo")
    split = np.load(ROOT / "data/team_split_ids.npz")
    train,val = split["train"],split["val"]
    core_indices=np.array(list(range(72))+list(range(144,186))+list(range(222,300))+list(range(378,417)))
    cases=[("structure_150",150,np.arange(X.shape[1])),
           ("structure_300",300,np.arange(X.shape[1])),
           ("core_structure_200",200,core_indices)]
    for name,iterations,indices in cases:
        folder=ROOT / "experiments" / name
        folder.mkdir(exist_ok=True)
        if (folder/"metrics.json").exists():
            continue
        settings=dict(max_iter=iterations,max_leaf_nodes=15,learning_rate=.08,
                      l2_regularization=1.,random_state=42,early_stopping=False,max_bins=128)
        print(f"Training {name}: {len(indices)} features",flush=True)
        start=time.perf_counter()
        clf=HistGradientBoostingClassifier(**settings).fit(X[train][:,indices],ds.labels[train])
        record={"name":name,"feature_kind":"structure","feature_indices":indices.tolist(),
                "settings":settings,"training_seconds":time.perf_counter()-start,
                "training_rows":len(train),"validation_rows":len(val),"split_seed":42,
                "feature_source_sha256":hashlib.sha256((ROOT/"structure_features.py").read_bytes()).hexdigest(),
                "validation":metrics(ds.labels[val],clf.predict(X[val][:,indices])),
                "demo":metrics(demo.labels,clf.predict(D[:,indices]))}
        with (folder/"development.pkl").open("wb") as f:
            pickle.dump({"classifier":clf,"indices":indices,"kind":"structure"},f)
        (folder/"metrics.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
        print(name, "validation",record['validation']['accuracy'],"demo",record['demo']['accuracy'],flush=True)
        print("validation recall",np.round(record['validation']['recall'],3),flush=True)
        print("demo recall",np.round(record['demo']['recall'],3),flush=True)


if __name__ == "__main__":
    main()
