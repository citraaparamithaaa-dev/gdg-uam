"""Export fitted numerical HGB trees into a portable NumPy/Numba inference package."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import importlib.util
import json
import pickle
import shutil
import sys
from pathlib import Path
import numpy as np
import numba
from competition_api import load_dataset

ROOT = Path(__file__).resolve().parent


def export(source, target, core=False):
    source = source.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as f:
        saved = pickle.load(f)
    clf, indices, kind = saved["classifier"], saved["indices"], saved["kind"]
    arrays = {k: [] for k in ("roots", "feature", "threshold", "left", "right", "value", "missing_left")}
    offset = 0
    for stage in clf._predictors:
        for predictor in stage:
            nodes = predictor.nodes
            if np.any(nodes["is_categorical"]):
                raise ValueError("This exporter supports numerical trees only")
            arrays["roots"].append(offset)
            arrays["feature"].extend(np.where(nodes["is_leaf"], -1, nodes["feature_idx"]).astype(np.int64))
            arrays["threshold"].extend(nodes["num_threshold"])
            arrays["left"].extend(nodes["left"].astype(np.int64) + offset)
            arrays["right"].extend(nodes["right"].astype(np.int64) + offset)
            arrays["value"].extend(nodes["value"])
            arrays["missing_left"].extend(nodes["missing_go_to_left"])
            offset += len(nodes)
    for key in arrays:
        dtype = np.float64 if key in ("threshold", "value") else np.int64
        arrays[key] = np.asarray(arrays[key], dtype=dtype)
    bias = np.asarray(saved.get("bias", np.zeros(8)),dtype=np.float64)
    package_kind = "core" if core else kind
    arrays.update(initial=clf._baseline_prediction.ravel().astype(np.float64)+bias,
                  indices=np.asarray(indices, dtype=np.int64), kind=np.asarray(package_kind))
    np.savez_compressed(target / "weights.npz", **arrays)
    shutil.copyfile(ROOT / "runtime_model.py", target / "model.py")
    if kind in ("signal", "structure"):
        shutil.copyfile(ROOT / "signal_features.py", target / "signal_features.py")
        if kind == "structure":
            shutil.copyfile(ROOT / "structure_features.py", target / "structure_features.py")
        if package_kind == "core":
            shutil.copyfile(ROOT / "core_features.py", target / "core_features.py")
    else:
        shutil.copyfile(ROOT / "examples/baseline_model.py", target / "baseline_features.py")
    (target / "requirements.txt").write_text(f"numpy=={np.__version__}\nnumba=={numba.__version__}\n", encoding="utf-8")
    # Check numerical export parity, using data excluded from the development fit.
    ds = load_dataset(ROOT / "data/team_val.npz")
    sys.path.insert(0, str(target.resolve()))
    spec = importlib.util.spec_from_file_location("exported_candidate", target / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.Model()
    expected, actual, max_error = [], [], 0.
    for i in range(1000):
        sample = ds[i]
        x = model.transform(sample)[indices]
        ref = clf._raw_predict(x[np.newaxis, :])[0]+bias
        fast = module.raw_scores(x, *model.trees)
        max_error = max(max_error, float(np.max(np.abs(ref - fast))))
        expected.append(int(np.argmax(ref)))
        actual.append(model.predict(sample))
    if expected != actual or max_error > 1e-9:
        raise AssertionError(f"Export changed predictions: max raw score error {max_error}")
    report = {"samples_checked": 1000, "prediction_matches": True, "max_raw_score_error": max_error,
              "trees": len(arrays["roots"]), "nodes": offset, "feature_kind": package_kind,
              "weights_bytes": (target / "weights.npz").stat().st_size,
              "source_model": str(source.relative_to(ROOT))}
    if package_kind == "core":
        # The pruned extractor must reproduce the structure features bit for bit.
        from structure_features import features as full_features
        from core_features import features as pruned_features
        mismatch = 0
        for i in range(1000):
            sample = ds[i]
            if not np.array_equal(full_features(sample.series)[indices],
                                  pruned_features(sample.series)[indices]):
                mismatch += 1
        if mismatch:
            raise AssertionError(f"Pruned extractor differs on {mismatch}/1000 samples")
        report["pruned_feature_matches"] = True
    (target / "export_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--core", action="store_true",
                        help="package the pruned core extractor (balanced_core_200)")
    args = parser.parse_args()
    export(args.source, args.target, core=args.core)
