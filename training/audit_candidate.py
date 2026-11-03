"""Recheck one selected model on a second stratified split, without demo-based tuning."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
import argparse
import json
import time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from competition_api import load_dataset
from train_candidates import metrics
from calibrate_bias import fit_bias

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    args = parser.parse_args()
    folder = ROOT / "experiments" / args.candidate
    record = json.loads((folder / "metrics.json").read_text())
    ids = np.asarray(record["feature_indices"])
    X = np.load(ROOT / f"data/features_{record['feature_kind']}_train.npy")[:, ids]
    ds = load_dataset(ROOT / "data/train_100k.npz")
    train, val = train_test_split(np.arange(len(ds)), test_size=.2, random_state=123,
                                 stratify=ds.labels)
    start = time.perf_counter()
    clf = HistGradientBoostingClassifier(**record["settings"]).fit(X[train], ds.labels[train])
    calibration=[]
    bias=np.zeros(8)
    if "class_bias" in record:
        calibration,val=train_test_split(val,train_size=5000,random_state=314,
                                         stratify=ds.labels[val])
        bias=fit_bias(clf._raw_predict(X[calibration]),ds.labels[calibration])
    prediction=np.argmax(clf._raw_predict(X[val])+bias,axis=1)
    result = {"candidate": args.candidate, "seed": 123,
              "training_rows": len(train), "validation_rows": len(val),
              "calibration_rows":len(calibration),"class_bias":bias.tolist(),
              "training_seconds": time.perf_counter() - start,
              "validation": metrics(ds.labels[val], prediction),
              "note": "Second random split checks split sensitivity; it is not a new untouched test set because development used other rows from the same training file."}
    (folder / "second_split_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
