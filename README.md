# GDG UAM · Machine Learning Competition

**Eight-class classification of multichannel time series**

A compact gradient-boosted classifier with **NumPy-only CPU inference**, trained weights included, and verification using the unmodified official evaluation harness.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/citraaparamithaaa-dev/gdg-uam/blob/main/verify_colab.ipynb)

[Run the submission](#run-the-submission) · [Method](#method) · [Detailed results](RESULTS.md) · [Colab evidence](evidence/colab/2026-11-12/) · [Official materials](https://github.com/GDG-UAM/ml-contest-materials)

## Verified results

The submitted model completed **three official-harness runs in Google Colab**, with identical accuracy and no evaluation failures. The archived model files match this repository byte for byte.

| Metric | Google Colab result |
| --- | ---: |
| Public demo accuracy | **82.25% — 329 / 400 correct** |
| Official score on the finite demo | **3.2924** |
| Prediction time for all 400 samples | **1.47–2.46 seconds** |
| Maximum model setup time | **1.44 seconds** |
| Successful official runs | **3 / 3** |
| Exceptions, invalid outputs, and timeouts | **0** |

**Environment:** Python 3.13.18, NumPy 2.5.4, CPU execution with numerical libraries restricted to one thread by the official harness. Prediction timing excludes installation, setup, and warmup. The model performs no runtime compilation.

The separate prediction-contract check also passed: all 400 predictions took **2.09–4.13 seconds**, across three repetitions. Both setup and whole-demo prediction met the **under-10-second target** in every recorded check. Timing varies with the assigned machine and load.

The [Colab verification record](evidence/colab/2026-11-12/VERIFICATION.json) documents archive provenance, file hashes, and individual run results. Six additional local official-harness runs, including a fresh GitHub clone, are recorded in [RESULTS.md](RESULTS.md).

## Run the submission

Use Python 3.12 or 3.13 and the pinned inference dependency in [`requirements.txt`](requirements.txt).

First follow the [official materials setup](https://github.com/GDG-UAM/ml-contest-materials#setup), including downloading `data/train_100k.npz` for the default evaluation command. Then, **from the directory containing the organizer's `harness.py`**, run:

```bash
git clone https://github.com/citraaparamithaaa-dev/gdg-uam.git submission-check
python -m pip install -r submission-check/requirements.txt
python harness.py evaluate --model submission-check/model.py
```

To reproduce the reported 400-sample demo results, download `data/test_demo.npz` through the official setup and run:

```bash
python harness.py evaluate --model submission-check/model.py --source file:data/test_demo.npz --runs 3
```

The organizer's materials supply `harness.py` and `competition_api.py`. The default command evaluates the training file; because the final model was fitted on those rows, its training-file accuracy is a compatibility check rather than an independent test result.

## Method

1. **Preprocess the signal.** Interpolate missing values and selected robust spike outliers; replace entirely missing channels with zeros.
2. **Extract 231 normalized features.** Describe distributions, local changes, autocorrelation, frequency content, and cross-channel behavior.
3. **Predict with gradient boosting.** Evaluate an exported 200-stage histogram gradient boosting model with up to 15 leaves per tree and calibrated class offsets.

The final weights were fitted on all **100,000 official training samples**. Feature extraction and tree traversal use vectorized NumPy operations. Weights load relative to `model.py`; each call returns one Python integer in `0`–`7`. Inference preserves read-only inputs and uses no metadata, network requests, GPU, training, or background prediction.

See [APPROACH.md](APPROACH.md) for preprocessing details, calibration, export, and experimental decisions.

## Submission files

| File or directory | Purpose |
| --- | --- |
| [`model.py`](model.py) | Top-level entry point defining `class Model` |
| [`weights.npz`](weights.npz) | Trained model parameters; approximately **0.67 MB** |
| [`fast_numpy_features.py`](fast_numpy_features.py) | Inference preprocessing and feature extraction |
| [`requirements.txt`](requirements.txt) | Pinned NumPy dependency |
| [`verify_colab.ipynb`](verify_colab.ipynb) | Self-contained model verification notebook |
| [`training/`](training/) | Training, export, and verification utilities |
| [`APPROACH.md`](APPROACH.md), [`RESULTS.md`](RESULTS.md) | Methodology, comparisons, and limitations |
| [`evidence/colab/2026-11-12/`](evidence/colab/2026-11-12/) | Archived Colab results and provenance |
| [`VERIFICATION.json`](VERIFICATION.json), [`MANIFEST.json`](MANIFEST.json) | Verification summary and SHA-256 file hashes |

All required inference files are committed. No file exceeds GitHub's 100 MB limit; Git LFS is not required. Earlier feature implementations remain available for development history and are not imported by the current entry point.

## Verify in Google Colab

1. Open the notebook using the badge above, or download [`verify_colab.ipynb`](verify_colab.ipynb) and upload it to Colab.
2. Select a **CPU runtime**, connect, and choose **Runtime → Run all**.
3. Review the integrity checks, prediction checks, and three official evaluation runs. The final cell downloads the submission package and results.

The notebook embeds the exact model, public demo, and unmodified official harness/API. An uploaded copy needs no GitHub sign-in or Google Drive mount. Package installation is separate from model setup.

## Reliability and interpretation

- **Input handling:** 22 numerical edge cases passed, including all-NaN signals, constants, sparse observations, and extreme float32 values. Input preservation, repeated predictions, and 100 metadata-independence checks also passed.
- **Runtime agreement:** the current implementation matched the previous compiled model on 5,000 additional training signals and 2,800 declared stress cases. This is observed class agreement, not bitwise equivalence on every possible input. See [the robustness report](evidence/latest/fast_setup_robustness.json).
- **Missing information:** severe data loss still reduces accuracy. The declared 64-sample outage test reached 57.75%; losing an entire channel reached 49.25%.
- **Evaluation limits:** the public demo was reused during development. Reported accuracy is a development result; hidden-test accuracy and rank remain unknown. The finite-demo score does not forecast the official hidden-stream score.
- **Startup tradeoff:** removing compilation shortens setup but slows individual predictions relative to the [earlier compiled runtime](https://github.com/citraaparamithaaa-dev/gdg-uam/tree/5e08d476da89d183489effd9eabc74ac7ee27671). Under the competition's fixed prediction-time budget, this can reduce the number of samples scored.

AI coding assistance contributed to implementation, experiments, and documentation. Supporting evidence and methodological limitations are included for review.
