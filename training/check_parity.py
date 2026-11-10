"""Verify the pruned core extractor reproduces the full structure features.

Compares, for every used feature slot, the value produced by
core_features.features against structure_features.features on real rows,
NaN-fuzzed rows and pathological inputs. A single differing value fails.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from competition_api import load_dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/test_demo.npz")
    parser.add_argument("--weights", default=str(HERE.parent / "weights.npz"))
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--fuzz", type=int, default=200)
    parser.add_argument("--report", default=str(HERE.parent / "evidence" / "pruning_parity.json"))
    args = parser.parse_args()

    from structure_features import features as full_features  # noqa: E402 (repo root on path)
    from core_features import features as pruned_features
    indices = np.load(args.weights, allow_pickle=False)["indices"]

    rng = np.random.default_rng(7)
    checked, mismatch = 0, 0

    def compare(arr):
        nonlocal checked, mismatch
        checked += 1
        a = full_features(arr)[indices]
        b = pruned_features(arr)[indices]
        if not np.array_equal(np.nan_to_num(a, nan=-9e99), np.nan_to_num(b, nan=-9e99)):
            mismatch += 1

    ds = load_dataset(args.source)
    for i in range(min(args.rows, len(ds))):
        compare(np.array(ds[i].series))
    for i in range(args.fuzz):
        arr = np.array(ds[i % len(ds)].series)
        for c in range(4):
            mask = rng.random(256) < rng.uniform(0, 0.5)
            arr[c][mask] = np.nan
        compare(arr)
    s0 = np.array(ds[0].series)
    edge = [np.zeros((4, 256), np.float32), np.ones((4, 256), np.float32),
            np.full((4, 256), np.nan, np.float32),
            np.repeat(np.array([[1.0], [2.0], [3.0], [4.0]], np.float32), 256, axis=1)]
    for arr in edge:
        compare(arr)

    t0 = time.perf_counter()
    for i in range(min(args.rows, len(ds))):
        pruned_features(ds[i].series)
    pruned_ms = (time.perf_counter() - t0) / min(args.rows, len(ds)) * 1e3
    t0 = time.perf_counter()
    for i in range(min(args.rows, len(ds))):
        full_features(ds[i].series)
    full_ms = (time.perf_counter() - t0) / min(args.rows, len(ds)) * 1e3

    report = {"source": args.source, "values_checked": int(checked * len(indices)),
              "rows_checked": int(checked), "mismatching_rows": mismatch,
              "full_extractor_ms": round(full_ms, 3), "pruned_extractor_ms": round(pruned_ms, 3)}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if mismatch:
        raise SystemExit(f"parity FAILED: {mismatch} mismatching rows")


if __name__ == "__main__":
    main()
