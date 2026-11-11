"""Compare a no-compilation runtime with the published model and stress inputs."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--reference', type=Path, required=True, help='Checkout of compiled runtime at commit 5e08d47')
parser.add_argument('--candidate', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--materials', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.materials.resolve()))
from competition_api import Sample, load_dataset


def load_model(folder, name):
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(name, folder / 'model.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, module.Model()


new_module, new = load_model(args.candidate.resolve(), 'fast_setup')
old_module, old = load_model(args.reference.resolve(), 'published_reference')
demo = load_dataset(args.materials / 'data/test_demo.npz')
train = load_dataset(args.materials / 'data/train_100k.npz')
rng = np.random.default_rng(20261112)
rows = rng.choice(len(train.labels), 5000, replace=False)
mismatches = []
maximum_feature_difference = 0.
maximum_score_difference = 0.
start = time.perf_counter()
for j, index in enumerate(rows):
    sample = train[int(index)]
    a = old.transform(sample)[old.indices]
    b = new_module.features(sample.series)[new.indices]
    before = old_module.raw_scores(a, *old.trees)
    after = new.scores(b)
    maximum_feature_difference = max(maximum_feature_difference, float(np.max(np.abs(a - b))))
    maximum_score_difference = max(maximum_score_difference, float(np.max(np.abs(before - after))))
    if before.argmax() != after.argmax():
        mismatches.append(int(index))
    if (j + 1) % 1000 == 0:
        print('Compared', j + 1, 'training signals', flush=True)

comparison = dict(samples=5000, seed=20261112, prediction_mismatches=len(mismatches),
                  mismatch_row_ids=mismatches, maximum_feature_difference=maximum_feature_difference,
                  maximum_raw_score_difference=maximum_score_difference,
                  elapsed_seconds=time.perf_counter() - start,
                  interpretation='Empirical prediction agreement, not bitwise equivalence or independent accuracy validation.')

def corrupt(array, condition):
    x = array.copy()
    if condition == 'missing_10pct':
        x[rng.random(x.shape) < .1] = np.nan
    elif condition == 'missing_30pct':
        x[rng.random(x.shape) < .3] = np.nan
    elif condition == 'outage_64':
        start = int(rng.integers(0, 193))
        x[:, start:start + 64] = np.nan
    elif condition == 'one_channel_missing':
        x[int(rng.integers(0, 4))] = np.nan
    elif condition == 'spikes_1pct':
        scale = np.nanstd(x, axis=1, keepdims=True) + 1e-6
        mask = rng.random(x.shape) < .01
        x = x + (mask * rng.choice([-1., 1.], x.shape) * scale * 10).astype(np.float32)
    elif condition == 'gain_and_offset':
        x = x * np.float32(10 ** rng.uniform(-2, 2)) + np.float32(rng.uniform(-10, 10))
    return x

conditions = ['natural', 'missing_10pct', 'missing_30pct', 'outage_64',
              'one_channel_missing', 'spikes_1pct', 'gain_and_offset']
stress = {}
for condition in conditions:
    original_answers, new_answers = [], []
    begin = time.perf_counter()
    for i in range(400):
        x = corrupt(demo[i].series, condition)
        before = x.copy()
        sample = Sample(x, {})
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            answer = new.predict(sample)
            assert answer == new.predict(sample)
        assert type(answer) is int and 0 <= answer <= 7
        assert np.array_equal(before, x, equal_nan=True)
        original_answers.append(old.predict(sample))
        new_answers.append(answer)
    stress[condition] = dict(samples=400, old_correct=int(np.sum(np.array(original_answers) == demo.labels)),
                            new_correct=int(np.sum(np.array(new_answers) == demo.labels)),
                            prediction_disagreements=int(np.sum(np.array(original_answers) != new_answers)),
                            errors=0, invalid_outputs=0, input_mutations=0, nondeterministic_outputs=0,
                            runtime_warnings=0, elapsed_seconds=time.perf_counter() - begin)
    print(condition, stress[condition], flush=True)

cases = []
for count in (0, 1, 9, 10, 19, 20, 128, 256):
    x = np.full((4, 256), np.nan, dtype=np.float32)
    x[:, :count] = rng.normal(size=(4, count)).astype(np.float32)
    cases.append(x)
for value in (0., 1., -1., np.finfo(np.float32).max, -np.finfo(np.float32).max, np.finfo(np.float32).tiny):
    cases.append(np.full((4, 256), value, dtype=np.float32))
x = np.zeros((4, 256), dtype=np.float32)
x[:, ::2] = np.finfo(np.float32).max
x[:, 1::2] = -np.finfo(np.float32).max
cases.append(x)
cases.append(rng.choice([np.inf, -np.inf, np.nan, 0., 1.], size=(4, 256)).astype(np.float32))
for x in cases:
    before = x.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        answer = new.predict(Sample(x, {}))
        assert answer == new.predict(Sample(x, {}))
    assert type(answer) is int and 0 <= answer <= 7
    assert np.array_equal(before, x, equal_nan=True)

report = dict(reference='Published Numba model at commit 5e08d47', comparison=comparison,
              stress_tests=stress, numerical_edge_cases=len(cases),
              note='Synthetic corruptions measure behavior under declared perturbations, not hidden-test accuracy. Infinity tests extend beyond the NaN input contract.')
path = args.out
path.write_text(json.dumps(report, indent=2))
print('Saved', path, flush=True)
