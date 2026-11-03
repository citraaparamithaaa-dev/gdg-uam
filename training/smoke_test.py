"""Check output type, missing data, immutable inputs, repeatability and metadata independence."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
from competition_api import Sample, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("../submission/model.py"))
    parser.add_argument("--report",type=Path,help="optional report outside the model repository")
    args = parser.parse_args()
    model_path = args.model.resolve()
    sys.path.insert(0, str(model_path.parent))
    spec = importlib.util.spec_from_file_location("tested_model", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    setup_start = time.perf_counter()
    model = module.Model()
    setup = time.perf_counter() - setup_start
    ds = load_dataset("data/team_val.npz")
    partial = ds[0].series.copy()
    partial[0, 20:60] = np.nan
    missing_channel = ds[0].series.copy()
    missing_channel[1] = np.nan
    arrays = [ds[0].series.copy(), np.zeros((4,256),np.float32),
              np.full((4,256),np.nan,np.float32), np.ones((4,256),np.float32),
              partial, missing_channel]
    edge_predictions = []
    for array in arrays:
        before = array.copy()
        sample = Sample(array, dict(ds[0].meta))
        prediction = model.predict(sample)
        assert type(prediction) is int and -1 <= prediction <= 7, prediction
        assert np.array_equal(array, before, equal_nan=True), "Input mutated"
        assert prediction == model.predict(sample), "Prediction not repeatable"
        edge_predictions.append(prediction)
    times = []
    metadata_checks = 0
    for i in range(1000):
        sample = ds[i]
        start = time.perf_counter()
        answer = model.predict(sample)
        times.append(time.perf_counter() - start)
        assert type(answer) is int and 0 <= answer <= 7
        if model.kind in ("signal", "structure") and i < 100:
            changed_meta = Sample(sample.series, {"sample_rate": 999., "group": "unseen", "level": -999.})
            assert model.predict(changed_meta) == answer, "Metadata changed a signal-only prediction"
            metadata_checks += 1
    report = {"edge_cases_passed": len(arrays), "edge_predictions": edge_predictions,
              "metadata_independence_checks": metadata_checks,
              "setup_seconds": setup, "prediction_samples": len(times),
              "mean_latency_ms": 1000*float(np.mean(times)),
              "p99_latency_ms": 1000*float(np.quantile(times,.99)),
              "max_latency_ms": 1000*float(max(times))}
    print(json.dumps(report, indent=2), flush=True)
    report_path=args.report or model_path.parent/"smoke_test_results.json"
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
