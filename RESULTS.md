# Model results — 12 November 2026

The current submission uses **NumPy-only inference with no runtime compilation**. It retains the selected trained weights and **329/400 (82.25%)** public-demo accuracy, while reducing model setup below the requested 10 seconds. Installation time is separate from model setup.

## Current runtime

The unmodified official harness evaluated the exact repository entry point in three fresh processes. Complete records are in [`evidence/latest/fast_setup_official_demo`](evidence/latest/fast_setup_official_demo).

| Run | Setup | All 400 predictions | Correct | Exceptions / invalid outputs / timeouts |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.286 s | 1.334 s | 329 | 0 / 0 / 0 |
| 1 | 0.294 s | 1.236 s | 329 | 0 / 0 / 0 |
| 2 | 0.294 s | 1.241 s | 329 | 0 / 0 / 0 |

A post-publication clone of commit `57a85f3` passed all 114 file hashes, matched the delivered notebook, and passed three additional official runs. Setup was 0.480–0.497 seconds and all 400 predictions took 1.830–1.961 seconds, again with 329 correct answers and no errors or timeouts. See [`fast_setup_fresh_clone`](evidence/latest/fast_setup_fresh_clone). These runs illustrate normal host-load variation; the model files were byte-identical.

All three runs returned the finite-demo score **3.2923757006**. Setup plus scored prediction time was below two seconds in each run; installation, warmup, producer work, and other process overhead are excluded from that sum.

The direct [contract check](evidence/latest/fast_setup_contract.json) passed three 400-prediction repetitions, 22 numerical edge cases, read-only input preservation, deterministic outputs, Python integer return values, and 100 metadata-independence checks. Socket connections were disabled during model import, setup, and prediction. The standalone notebook payload was also restored and executed independently; see [its verification record](evidence/latest/fast_setup_notebook_payload.json).

**Colab execution remains unverified.** Local times cannot certify another machine. The notebook enforces setup below 10 seconds and all 400 predictions below 10 seconds when run on the assigned CPU.

## Startup and throughput tradeoff

The prior compiled runtime, available at commit `5e08d47`, recorded setup up to 33.96 seconds and 0.48–0.75 seconds to predict all 400 samples. The current runtime avoids compilation and requires only NumPy, but takes longer per prediction. The official score depends on the number and quality of predictions within a fixed 60-second prediction-time budget, so lower setup time does not by itself improve competition score. No hidden-stream comparison or winning rank has been established.

Earlier compiled-runtime evidence is retained in [`evidence/latest/official_demo`](evidence/latest/official_demo) and [`compiled_runtime_verification.json`](evidence/latest/compiled_runtime_verification.json). Those timing records describe the previous runtime.

## Robustness and prediction agreement

The NumPy runtime matched the prior compiled model's class predictions on 5,000 additional training signals sampled with seed `20261112`. Maximum retained-feature difference was approximately `4.09e-14`; maximum raw-score difference was `0.490`, because tiny feature changes can cross tree thresholds. Zero observed class disagreements do not establish bitwise equivalence or agreement on every future input.

The following checks each used all 400 demo signals with declared synthetic perturbations. Both runtimes returned the same classes in all 2,800 cases, with zero exceptions, invalid outputs, runtime warnings, input mutations, or inconsistent repeated answers.

| Condition | Correct / 400 | Accuracy |
| --- | ---: | ---: |
| Natural demo | 329 | 82.25% |
| Randomly missing 10% of values | 310 | 77.50% |
| Randomly missing 30% of values | 312 | 78.00% |
| Contiguous 64-sample outage in every channel | 231 | 57.75% |
| One entire channel missing | 197 | 49.25% |
| Added spikes at 1% of positions | 321 | 80.25% |
| Common positive gain and offset changes | 329 | 82.25% |

These perturbations measure specified behavior rather than hidden-test accuracy. Robust input handling does not prevent accuracy loss when discriminating information is missing. Numerical edge cases include NaN-only and constant signals, sparse finite observations, extreme float32 magnitudes, and infinity handling beyond the official NaN contract.

Full evidence: [`fast_setup_robustness.json`](evidence/latest/fast_setup_robustness.json). To reproduce the comparison, first obtain a separate checkout of the compiled runtime at commit `5e08d47`, then run:

```bash
python training/check_runtime_robustness.py --reference PATH_TO_COMPILED_CHECKOUT --candidate . --materials PATH_TO_OFFICIAL_MATERIALS --out robustness.json
```

The comparison needs NumPy and Numba to run the older reference, plus the official training and demo files. Inference with the current submission requires NumPy alone.

## Model selection and data preparation

| Final candidate | Correct / 400 | Demo accuracy | Finite-demo score |
| --- | ---: | ---: | ---: |
| Selected weights, current NumPy runtime | 329 | 82.25% | 3.2924 |
| CatBoost | 317 | 79.25% | 2.7857 |
| CatBoost/HGB blend | 328 | 82.00% | 3.0214 |
| XGBoost | 324 | 81.00% | 2.7398 |
| Covariate weighting | 326 | 81.50% | 3.1489 |

The data audit covered all 100,000 training and 400 demo signals. It found no duplicate signals, invalid labels, infinities, empty channels, or constant channels. Missing values and robust tails were more frequent in the demo.

Three consistently applied cleaning pipelines were compared on fixed training-file partitions: interpolation reached 87.67% natural validation accuracy, Hampel filtering 87.62%, and Savitzky–Golay smoothing 86.81%. Additional experiments compared three CatBoost configurations, XGBoost, five declared blend weights, and bounded covariate weighting learned from unlabeled demo signals. XGBoost reached 88.41% natural validation accuracy and the blend 88.29%, but neither improved the public demo. The selected tree weights were refitted on all 100,000 training rows.

The selected inference preprocessing retains the established spike interpolation; its original training features were unclipped. That mismatch and the repeated use of the demo are development limitations. Final-refit accuracy on the training file is not independent validation. No hidden evaluation data was accessed, and no hard accuracy ceiling has been demonstrated.

## Reproduction and submission

The current entry point, trained weights, inference dependency pins, notebook, and file hashes are committed. The notebook embeds the exact model, public demo, and unmodified official evaluator/API, checks integrity, runs prediction checks, and evaluates three fresh processes. It downloads the results and a submission archive without requiring a Drive mount.

Training and historical model comparisons require additional development dependencies, including scikit-learn, Numba, CatBoost, XGBoost, and pandas. See [`training/`](training/) and the recorded evidence for the original workflows. These packages are not required to run the current exported model.

Organizer clone access remains unverified. Registration is complete; team and member names will be provided separately. No submission email has been sent. The supplied slides list 12 November at 23:59 without a timezone, so the organizer's deadline interpretation should govern submission.

Official rules and evaluator: [GDG-UAM/ml-contest-materials](https://github.com/GDG-UAM/ml-contest-materials).
