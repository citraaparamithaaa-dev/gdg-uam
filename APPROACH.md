# Method and evidence

The selected classifier uses 231 scale-normalized signal features and a 200-stage histogram gradient boosting model with up to 15 leaves per tree. The numerical tree parameters and class offsets are stored in `weights.npz`. The offsets were calibrated using 5,000 training-file holdout rows; 15,000 other holdout rows were retained for development checks. The final classifier was refitted on all 100,000 official training rows.

## Preprocessing and features

Missing values are interpolated, and entirely missing channels become zeros. At inference, finite values more than 4.5 scaled median absolute deviations from the channel median are replaced by interpolation between retained neighbors. Earlier experiments selected this despiking rule using the public demo. The original model was fitted on unclipped features; this mismatch is retained because the consistently cleaned alternatives did not improve the measured result.

Features describe the first two channels' distributions, local changes, autocorrelation and spectra; cross-channel correlations; and amplitude/frequency behavior in the first two channels and their shared component. Metadata and absolute level/scale are excluded.

## Current CPU runtime: setup below 10 seconds

The current `model.py` and `fast_numpy_features.py` use NumPy alone. They load the same trained numerical weights, process only the retained features, transform 21 spectral windows together, and traverse the trees using vectorized arrays. There is no runtime compilation or dependency on Numba, SciPy, or scikit-learn. Finite and nonfinite values are handled in private float64 copies, preserving read-only inputs.

Three official demo runs of this runtime had setup times of 0.286–0.294 seconds and prediction times of 1.236–1.334 seconds for all 400 samples. Each returned 329 correct predictions with no exceptions, invalid outputs, or timeouts. Package installation is outside these timings.

A comparison against the previous compiled model found zero class disagreements on 5,000 additional training signals and all 2,800 declared stress cases. Small feature differences still caused some raw-score differences, so this is empirical class agreement rather than bitwise equivalence. Stress accuracy was 77.50% with 10% random missing values, 78.00% with 30% random missing values, 57.75% with a 64-sample outage, 49.25% with a missing channel, 80.25% with 1% added spikes, and 82.25% after gain/offset changes. These are fixed diagnostic perturbations, not estimates of hidden-test performance.

The tradeoff is slower per-sample prediction than the earlier compiled runtime. Under the official fixed prediction-time budget, that can reduce hidden-stream throughput and score. The new runtime meets the requested sub-10-second setup target, while the previous version remains in Git history at commit `5e08d47`.

## Earlier compiled runtime

The earlier runtime's `fused_features.py` groups the feature reductions into compiled stages, leaving FFTs and the two small matrix products in NumPy. Keeping matrix products outside compiled functions avoids an optional SciPy BLAS dependency. `core_features.py` supplies the reductions and missing-value/spike handling. Its entry point compiled on a zero sample during setup, using Numba optimization level 1. These helpers are retained for development history; the current entry point does not import them.

The optimized feature stages have floating-point differences from the earlier implementation. A 5,000-sample comparison had zero prediction disagreements, while raw-score differences occurred when tiny feature changes crossed tree thresholds. This is empirical prediction agreement, not bitwise parity or a guarantee of equality on every input.

## Completed comparison

The data audit checked all training and demo rows. There were no duplicate signals, invalid labels, infinities, empty channels or constant channels. Missing values were 0.7257% of training values and 1.3796% of demo values. Values beyond the robust tail threshold were about three times more frequent in the demo; that threshold does not establish that every marked value is an error.

Three pipelines applied consistently before fitting and inference were compared with identical model settings and fixed splits. Interpolation reached 87.67% natural validation accuracy, Hampel filtering 87.62%, and light Savitzky-Golay smoothing 86.81%. The spike filter improved a declared synthetic stress check but did not improve the public demo.

Three CatBoost configurations, a regularized XGBoost model, five fixed CatBoost/HGB mixture weights, and bounded covariate weighting were then compared. Covariate weights used unlabeled demo signals, not demo class labels. Selection and calibration used training-file partitions; the demo remained a repeatedly reused diagnostic. XGBoost reached 88.41% natural validation accuracy and the selected blend 88.29%, but neither improved the demo. Their numerical exports were checked against their native evaluators on 1,000 feature vectors.

| Final candidate | Demo accuracy | Finite-demo score |
|---|---:|---:|
| Selected fast model | 82.25% | 3.2924 |
| CatBoost | 79.25% | 2.7857 |
| CatBoost/HGB blend | 82.00% | 3.0214 |
| XGBoost | 81.00% | 2.7398 |
| Covariate weighting | 81.50% | 3.1489 |

The earlier compiled model had the shortest median whole-demo prediction time in that matched comparison. The finite-file score is not a forecast of the official 60-second hidden stream. Historical evidence remains in `evidence/`; the latest comparisons and fresh official runs are in `evidence/latest/`.

## Limits

The public demo has only 400 samples and has been reused. The holdout informed development decisions, and final training-file accuracy is resubstitution accuracy. No hidden evaluation set was accessed, no hard accuracy ceiling has been demonstrated, and no winning rank is guaranteed.

The supplied Colab result archive matched all nine embedded notebook input files and the repository's four runtime files. On Python 3.13.18 with NumPy 2.5.4, three official runs each returned 329/400 correct, with setup at 0.845–1.434 seconds and all 400 predictions at 1.470–2.458 seconds. There were zero exceptions, invalid outputs, or timeouts. Separate direct checks took 2.087–4.127 seconds for all 400 samples and passed all 22 numerical edge cases. This verifies the supplied Colab run, not performance on every assigned machine. See [the archived Colab evidence](evidence/colab/2026-11-12/VERIFICATION.json).
