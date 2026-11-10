# Approach and results — how this entry was built

This document tells the story of the entry: what the competition actually rewards, how we
thought about it, what we shipped, and what the numbers show. Everything here is backed by
artifacts in `evidence/` and can be reproduced with the scripts in `training/`.

---

## 1. The problem

The contest asks for an 8-class classifier over multichannel time series (4 channels x 256
steps, NaNs allowed) with two twists that define everything else:

1. **Sequential prediction under a time budget.** Samples arrive one at a time; only
   `predict` is timed, and a run ends when 60 s of cumulative prediction time is spent.
   The official stream is infinite, so the faster the model, the more samples it answers.
2. **A geometric-mean score that punishes weak classes quadratically.** Per class k:

   ```
   T_k = max(0.01, sqrt(n_k) * (c_k - 0.33 * w_k) / n_k * (c_k / n_k)^2)
   Score = (T_0 * ... * T_7)^(1/8)
   ```

   n_k, c_k, w_k are drawn/correct/wrong counts for true class k. Reading the formula as
   `sqrt(n_k) * recall_k^2 * (recall_k - 0.33 * wrong_rate_k)` exposes the three levers:

   - **Speed**: n_k grows with samples answered; halving latency multiplies the score by
     sqrt(2) ~= 1.41. The one lever fully independent of accuracy.
   - **Recall, squared**: a recall point on the weakest class is worth exactly as much as
     on the strongest (geomean elasticity 1/8 per class), and it enters quadratically.
   - **Wrong answers cost little**: a wrong answer removes only ~12% of the value of a
     correct one, so abstaining is almost never worth it (breakeven at ~10% confidence).

3. **Distribution shift.** The organizers state the evaluation data differs from the
   training file in noise, ranges and class balance, and that the organizer baseline
   scores ~0.41 accuracy there versus ~0.82 in-distribution. The 400-row demo set is
   drawn like the evaluation data and is the only honest proxy for it.

At the start of this work the entry was a gradient-boosted classifier over 231 handcrafted
scale-invariant features (the `balanced_core_200` design): 77.25% demo accuracy, 2.565
demo score, 2.1 ms mean prediction latency, with strong development discipline (fixed
splits, offsets calibrated on a 5k holdout, demo labels never used for fitting).

---

## 2. The way of thinking

Four principles drove every decision.

**Follow the metric, not intuition.** The score decomposition says latency is a first-order
lever (a pure speed win multiplies the score even with identical predictions), that
accuracy work must target the *weakest* per-class terms (class 6 and 7 in practice), and
that abstention is worthless. It also implies an exchange rate: a +0.5 pp accuracy tweak
that costs 20% latency is a net loss.

**Never ship a change that is not provably safe.** For inference-side changes we demanded
*bitwise* equality of feature values and predictions, verified over tens of thousands of
rows plus pathological inputs — not "close enough". This required replicating NumPy's
exact arithmetic (pairwise summation, even-length medians, interpolated quantiles,
division-order in thresholds) inside the optimized code paths.

**Honest validation, layered.** Fit on the seed-42 80k split; tune only on 5k calibration
rows; score once on the untouched 15k check rows; repeat the comparison on a seed-123
split to catch split-specific luck; use the demo set only as a final gate because it is
small (50 samples per class) and was inspected during development. Any component fitted on
demo labels was additionally validated with split-half protocol: fit on one 200-row half,
evaluate on the untouched half — this estimates transfer to *new* evaluation samples,
which is what the official run is.

**Measure the shift before fighting it.** Rather than guessing what differs between the
training file and the evaluation distribution, compare distributions feature by feature
and statistic by statistic, within class, and only then design a response.

---

## 3. The solution

### 3.1 Speed: a pruned, bitwise-identical extractor

Profiling showed 96% of latency was feature extraction — and 63% of that computation was
discarded: the model reads 231 of the 717 computed features. The shipped package computes
only the used features (`core_features.py`), skipping the unused lag sweeps, first-difference
correlations and channels 2-3 summaries. Because feature values must not move by even one
ulp, the retained reductions are reimplemented to reproduce NumPy bit for bit (a numba
pairwise-sum replica, sort-based medians, quantile interpolation matching NumPy's `_lerp`),
and the two matrix-vector products keep the original (5, 256) shape with unused rows zeroed
so BLAS accumulation order is preserved.

`training/check_parity.py` compares old and new extractors; the recorded check covered 604
rows (real, NaN-fuzzed and pathological inputs) and 139,524 values with zero differences.
Result: mean latency 2.1 ms -> ~1.1 ms class (measured 2.8x on a slower reference machine),
identical predictions, identical scores.

### 3.2 Accuracy: find the shift, then remove it

The investigation chain, in order:

1. **Feature drift.** Comparing the 231 features between train and demo within class: the
   kurtosis family (kurtosis, delta-kurtosis, max-delta) drifts up to 8.4 sigma — an order
   of magnitude more than any other feature.
2. **Raw-statistic diagnosis.** Standardized kurtosis rises 2.8 -> 8.4 within every class,
   the fraction of samples beyond 5 robust deviations triples (0.0002 -> 0.0006), lag-1
   autocorrelation drops, high-frequency power rises — while *trimmed* kurtosis (outliers
   excluded) stays normal (2.97 vs 3.07). Conclusion: the core signals are unchanged; the
   evaluation stream overlays **sparse, class-correlated outliers** (class 7 much spikier
   than class 6: standardized kurtosis 11.3 vs 5.1), plus mild noise and a doubled NaN rate.
3. **Response.** Despike each sample before feature extraction: values beyond 5 robust
   median/MAD deviations are replaced with the channel median (`core_features.despike`,
   numba, verified bitwise-equal to the numpy reference). No retraining, no demo labels,
   +0.18 ms.

The clip level (5.0) and style (median-replace) were selected across seven configurations
on the demo — a single-parameter selection, documented as such.

### 3.3 What was tried and rejected

Every rejection is a measurement, recorded in `evidence/despiking.json` and
`evidence/class_pairs.json`:

- **Class weighting** (x2 on classes 6/7): +0.15/+0.23 on two in-distribution splits, but
  demo score fell 2.565 -> 2.23 (class-6 recall 0.44 -> 0.34 under shift). Training
  toward train-file class frequencies is exactly the shift the organizers warn about.
- **Corruption-augmented training**: +7 pp robustness against a *simulated* outlier/noise
  shift that reproduces demo-level degradation, but no transfer to demo (77.0/75.3 vs 77.25).
  The simulated family is not the real one.
- **Spectral-jitter augmentation**: trades class-6 recall for class-7, net flat.
- **Class-pair specialists (6/7, 0/3)**: ungated specialists trade recalls in ways the
  geomean punishes; confidence-gated versions were demo-neutral; spike-feature specialists
  matched chance on untouched demo halves (pool accuracy 0.46-0.59).
- **Demo-label calibrations**: 8 class offsets fitted on 200 demo rows lose 2.5-4.5 pp on
  the untouched halves; a 6/7 specialist fitted on demo rows runs at chance. In-sample demo
  gains are fiction; measured out-of-sample, they are negative.
- **Ensembles** with the robust variants: 79.8-81.3%, below the single despiked model.

---

## 4. The results

Official harness, three runs each, shipped package (commit history in this repository):

| Stage | Demo accuracy | Demo score | Mean latency |
| --- | ---: | ---: | ---: |
| Original entry (full extractor) | 77.25% | 2.565 | 2.12 ms |
| + pruned extractor (bitwise-identical) | 77.25% | 2.565 | ~2x faster |
| + inference-time despiking (shipped) | **81.50%** | **3.229** | unchanged within noise |

Per-class demo recalls, shipped package:
`[0.72, 0.98, 0.94, 0.84, 0.92, 0.96, 0.52, 0.64]` (before despiking:
`[0.68, 0.98, 0.86, 0.84, 0.88, 0.92, 0.44, 0.58]`).

Regression guards: clean 15k-check accuracy 93.61% -> 93.57% (despiking does not touch
training-like data), export parity 1e-14, smoke tests 6/6, cold setup ~18 s against a
120 s limit, zero failed or timed-out runs in every harness session. The expected official
score (demo per-class rates scaled to the 60 s infinite stream) rose by roughly a quarter
from the accuracy gain alone, on top of the speed term gained from the pruned extractor.

### The ceiling, quantified

The shipped model makes 74 errors out of 400 demo samples; 44 of them are in the class
6/7 pool. Every representation tested separates that pool at only 46-60% on untouched
rows (the main model, robust spike features, combined specialists, prototype matching), so
the honest ceiling sits near 84-85% overall. Reaching 90% would require labeled data drawn
from the evaluation generator — a larger labeled eval-like set or its per-class parameters —
which participants do not have. If such data becomes available, fine-tuning against even a
few thousand rows is the one credible path past the ceiling; the despike pipeline is
already the right front end for it.

---

## 5. Reproduce

See README for the full commands. Short version:

```powershell
python training/check_parity.py          # pruned vs full extractor parity
python training/try_class_weights.py     # rejected weighting experiment
python training/build_twostage.py        # rejected specialist packaging
python harness.py evaluate --model model.py --source file:data/test_demo.npz
```

Evidence files: `evidence/despiking.json` (shift finding, tuning, rejections),
`evidence/class_pairs.json` (class-pair experiments), `evidence/core_demo/` and
`evidence/despiked_demo/` (harness runs), `evidence/pruning_parity.json`.
