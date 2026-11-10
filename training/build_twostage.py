"""Build a two-stage package: main classifier + confidence-gated 6/7 specialist."""
import pickle
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "submission"))

TAU, MARGIN = 0.5, 0.25

main_arrays = {}
with np.load(HERE / "submission/weights.npz", allow_pickle=False) as p:
    for k in p.files:
        main_arrays[k] = p[k]

with (HERE / "candidates/main_dev.pkl").open("rb") as f:
    main = pickle.load(f)
with (HERE / "candidates/spec_6_7.pkl").open("rb") as f:
    spec = pickle.load(f)["specialist"]

# The shipped weights.npz holds the FINAL refit; the two-stage candidate keeps
# that classifier and adds the specialist trees tuned on the development fit.
def pack(clf):
    arrays = {k: [] for k in ("roots", "feature", "threshold", "left", "right", "value", "missing_left", "initial")}
    offset = 0
    for stage in clf._predictors:
        for predictor in stage:
            nodes = predictor.nodes
            arrays["roots"].append(offset)
            arrays["feature"].extend(np.where(nodes["is_leaf"], -1, nodes["feature_idx"]).astype(np.int64))
            arrays["threshold"].extend(nodes["num_threshold"])
            arrays["left"].extend(nodes["left"].astype(np.int64) + offset)
            arrays["right"].extend(nodes["right"].astype(np.int64) + offset)
            arrays["value"].extend(nodes["value"])
            arrays["missing_left"].extend(nodes["missing_go_to_left"])
            offset += len(nodes)
    for key in arrays:
        dtype = np.float64 if key in ("threshold", "value", "initial") else np.int64
        arrays[key] = np.asarray(arrays[key], dtype=dtype)
    arrays["initial"] = np.atleast_1d(clf._baseline_prediction.ravel().astype(np.float64))
    return arrays

spec_trees = pack(spec)
out = dict(main_arrays)
for k, v in spec_trees.items():
    out["spec_" + k] = v
out["spec_threshold"] = np.asarray(TAU)
out["spec_margin"] = np.asarray(MARGIN)
out["spec_pair"] = np.asarray([6, 7], dtype=np.int64)
target = HERE / "submission_v3"
target.mkdir(exist_ok=True)
np.savez_compressed(target / "weights.npz", **out)
for name in ("model.py", "core_features.py", "signal_features.py", "structure_features.py", "requirements.txt"):
    src = HERE / "submission" / name
    (target / name).write_bytes(src.read_bytes())
print(f"built {target} (tau={TAU}, margin={MARGIN})")
