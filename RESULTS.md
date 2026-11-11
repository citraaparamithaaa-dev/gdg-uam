# Final model selection â€” 11 November 2026

The best verified demo accuracy remains **329/400 (82.25%)**. No completed new candidate improved it. The selected package uses the existing trained weights with the faster feature implementation.

| Candidate | Correct | Demo accuracy | Official finite-demo score |
|---|---:|---:|---:|
| Selected fast model | 329/400 | 82.25% | 3.2924 |
| CatBoost | 317/400 | 79.25% | 2.7857 |
| CatBoost/HGB blend | 328/400 | 82.00% | 3.0214 |
| XGBoost | 324/400 | 81.00% | 2.7398 |
| Covariate weighting | 326/400 | 81.50% | 3.1489 |

The same 400 samples were used for the demo comparisons. These results are development evidence because this small demo has been reused. The scores above describe a finite 400-sample file, not the official 60-second hidden stream.

## What was tested

The data-quality audit covered all 100,000 training samples and all 400 demo samples. Three consistent cleaning pipelines were compared. Interpolation alone gave 87.67% natural validation accuracy, Hampel filtering 87.62%, and Savitzky-Golay smoothing 86.81%. There were no duplicate signals, invalid class labels, infinities, empty channels or constant channels. Missingness and robust tails were greater in the demo.

The additional campaign compared three CatBoost configurations, XGBoost, bounded covariate weighting learned from unlabeled demo signals, and five declared CatBoost/HGB mixture weights. Models used the same 80,000 training / 5,000 calibration / 15,000 validation split. The selected classifiers were refitted on all 100,000 training rows. XGBoost achieved 88.41% natural validation accuracy and the blend 88.29%, but their demo accuracies were lower. The new covariate-weighting method did not use demo class labels to estimate weights.

The new boosting models were exported to numerical arrays and tested against their native evaluators on 1,000 feature vectors. Their runtimes need no CatBoost or scikit-learn. The final selected fast entry needs only NumPy and Numba. It retains the established global despiking at inference; its original training features were unclipped. That preprocessing mismatch and the reused demo are disclosed rather than treated as independent generalization evidence.

## Runtime and requirements

The selected model's latest direct check passed three whole-demo repetitions below 10 seconds, valid Python integer returns, six pathological/missing-data cases, unchanged inputs, deterministic repeated predictions, and 100 metadata-independence checks. Socket connections were disabled during model import, setup and prediction. The fresh packaged copy took at most 71.35 seconds for setup across three official runs, within the 120-second limit.

The official evaluator runs models in fresh processes, limits numerical libraries to one CPU thread, permits 120 seconds for setup, and stops scoring at 60 cumulative prediction seconds. Calls over 0.5 seconds count as wrong and calls over 0.75 seconds can terminate a run. The official source checksum is `bb362375823362ac6ddde3ec170e1d091fd5988d208f35b9352c8c30e327ca82`; the official harness is unmodified.

The final notebook embeds the exact runtime files, trained weights, public demo, unmodified organizer evaluator/API, and integrity hashes. It installs the model's dependencies in a separate folder, then runs prediction safety checks and three official evaluation processes. It downloads the results and a submission archive. It does not need a GitHub login or a Google Drive mount. Its payload is also unpacked and executed locally to verify that it is self-contained.

**Colab execution is not yet verified.** Local times cannot certify the speed on a different Colab CPU. Run the delivered notebook on a CPU runtime and keep its resulting summary.

## Submission administration

The user confirmed that registration is complete. Team and member names will be provided later. Organizer access to the GitHub repository remains unverified, no submission email has been sent, and the supplied official slides list 12 November at 23:59 without specifying a timezone. This repository update contains the selected runtime and notebook. Submission to the organizers is a separate step.

Earlier documents claiming an 84â€“85% accuracy ceiling or presenting final-refit training-file accuracy as independent validation should not be used as evidence. No such ceiling or independent hidden-test result has been demonstrated.

## Reproduction

Training requirements include the pinned project dependencies plus CatBoost 1.2.10, XGBoost 3.4.1 and pandas 3.1.1. The original selected model training scripts remain in `training/`. The additional rejected trials are recorded in `evidence/latest/`. Their workspace runners are not required for inference or submission.

Official rules and evaluator: https://github.com/GDG-UAM/ml-contest-materials

## Final delivered package checks

The exact selected folder passed three fresh official runs: **329/400 (82.25%)**, score **3.2923757**, zero invalid outputs, exceptions or timeouts. Maximum fresh setup was **71.35 seconds**. In the final five interleaved timing repetitions, all 400 predictions took a median **1.381 seconds**, with every repetition below 10 seconds.

The exact notebook payload passed syntax and embedded-file hash checks, was restored into an independent folder, and reproduced 329 correct predictions. Its independent local check passed setup under 120 seconds, all three 400-prediction repetitions under 10 seconds, pathological NaN/constant inputs, unchanged inputs, repeatability, metadata independence and disabled socket connections. These are local checks; the notebook must still be run in Colab to certify that machine.

## Published runtime startup improvement

The final publication adds Numba optimization level 1, leaving numerical thread limits unchanged. This reduces cold compilation cost. The earlier matched benchmark predates this setting.

The updated runtime reproduced 329/400 (82.25%). Three fresh official runs passed with zero exceptions, invalid outputs or timeouts; maximum setup was 33.96 seconds. The latest direct whole-demo checks had median prediction time 0.694 seconds. The rebuilt notebook's exact payload was also tested independently. Colab execution remains unverified.
