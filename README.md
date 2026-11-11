# GDG UAM · Time-Series Classification

A CPU-based submission for the **GDG UAM Machine Learning Competition**, classifying four-channel time series into eight classes.

The submitted model combines robust signal preprocessing, 231 normalized features, and histogram gradient boosting. Trained weights are included; evaluation requires no retraining.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/citraaparamithaaa-dev/gdg-uam/blob/main/verify_colab.ipynb)

[Evaluation instructions](#evaluate-a-fresh-clone) · [Method](APPROACH.md) · [Results](RESULTS.md) · [Verification evidence](VERIFICATION.json) · [Official competition materials](https://github.com/GDG-UAM/ml-contest-materials)

## Submission at a glance

| Item | Details |
| --- | --- |
| Entry point | [`model.py`](model.py), defining `class Model` |
| Trained model | [`weights.npz`](weights.npz), approximately 0.67 MB |
| Dependencies | [`requirements.txt`](requirements.txt): NumPy only |
| Tested Python version | Python 3.12 |
| Execution | CPU inference; one numerical thread under the official harness |
| Prediction | One Python integer from `0` to `7` per sample |
| Training data | All 100,000 examples in the official training file |

All required weights and helper files are committed to this repository. No file exceeds GitHub's 100 MB limit, and Git LFS is not required.

## Evaluate a fresh clone

Use the **unmodified official harness** from the organizer's materials repository. Follow its [setup instructions](https://github.com/GDG-UAM/ml-contest-materials#setup), including downloading `data/train_100k.npz` for the default evaluation command.

Starting from the directory containing the organizer's `harness.py`:

```bash
git clone https://github.com/citraaparamithaaa-dev/gdg-uam.git submission-check
python -m pip install -r submission-check/requirements.txt
python harness.py evaluate --model submission-check/model.py
```

The last line is the competition's submission-check command. With the current public harness, it evaluates the training file by default. Because the final model was fitted on that file, its accuracy there is a compatibility check rather than an independent measure of generalization.

### Public demo check

For the 400-sample demo, download `data/test_demo.npz` as described in the official setup and run:

```bash
python harness.py evaluate --model submission-check/model.py --source file:data/test_demo.npz --runs 3
```

This is the command used for the reported demo results. The official harness and `competition_api.py` are supplied by the organizer; run these commands from that materials directory so both are available.

## Measured results

Three runs of the official harness completed successfully using the selected runtime. A separate fresh clone of the published GitHub commit also passed three demo runs.

| Metric | Recorded result |
| --- | ---: |
| Public demo accuracy | **82.25% — 329 / 400** |
| Official score on the finite demo | 3.2924 |
| Total model prediction time for all 400 samples | 1.24–1.34 seconds per run |
| Maximum setup time across the three recorded runs | **0.30 seconds** |
| Exceptions, invalid predictions, and timeouts | 0 |

The table's timings come from the committed [official evaluator logs](evidence/latest/fast_setup_official_demo). Prediction time excludes dependency installation, model setup, and warmup. The current runtime performs **no JIT compilation**. Both setup and all 400 predictions were below 10 seconds in every recorded run; the official setup allowance is 120 seconds. Timing varies by machine; these measurements were made locally.

Compared with the previous compiled runtime, setup fell from 33.96 seconds to 0.30 seconds. Per-sample prediction takes longer: the previous runtime processed all 400 samples in 0.48–0.75 seconds. Since official scoring uses a fixed prediction-time budget, this startup improvement can reduce the number of samples scored on the hidden stream. The earlier compiled version remains available at [commit 5e08d47](https://github.com/citraaparamithaaa-dev/gdg-uam/tree/5e08d476da89d183489effd9eabc74ac7ee27671).

### Robustness checks

The new runtime matched the previous model on **5,000 additional training signals** and **2,800 demo-based stress cases**, covering random missing values, contiguous outages, a missing channel, sparse spikes, and changes in gain and offset. All checks completed without invalid outputs, input modification, or inconsistent predictions. Another **22 numerical edge cases** passed, including all-NaN inputs, constants, and extreme float32 values. Full results and the reproducible comparison are in [the robustness report](evidence/latest/fast_setup_robustness.json) and [`training/check_runtime_robustness.py`](training/check_runtime_robustness.py).

These checks establish tested behavior, not immunity to data loss. Accuracy fell to 57.75% with a 64-sample outage and 49.25% with one entire channel missing. Neither runtime can recover information that is absent. Prediction agreement is empirical; small floating-point differences can still cross tree thresholds on other inputs.

**Interpretation:** the public demo has been reused during development, including preprocessing selection. Its accuracy is a development result, not an untouched test estimate. The finite-demo score does not predict the score on the organizers' hidden evaluation stream. Colab execution and hidden-test performance have not been verified here.

See [RESULTS.md](RESULTS.md) for candidate comparisons and [APPROACH.md](APPROACH.md) for the method, data checks, and limitations.

## Method

1. **Preprocess each signal.** Interpolate missing values, replace entirely missing channels with zeros, and apply the selected robust spike filter.
2. **Extract 231 features.** Describe normalized distributions, local changes, autocorrelation, spectra, and cross-channel behavior.
3. **Classify with gradient boosting.** Evaluate the exported 200-stage histogram gradient boosting model, with up to 15 leaves per tree, and calibrated class offsets.

The runtime loads weights relative to `model.py`, handles read-only inputs, and returns one prediction per sample. It uses no metadata, network requests, GPU, training, or background prediction during inference. Vectorized NumPy features and tree traversal avoid runtime compilation. Neither Numba nor scikit-learn is required for inference.

## Verify in Google Colab

1. Open [the Colab notebook](https://colab.research.google.com/github/citraaparamithaaa-dev/gdg-uam/blob/main/verify_colab.ipynb), or download [`verify_colab.ipynb`](verify_colab.ipynb) and upload it to Colab.
2. Select a **CPU runtime**, connect, and choose **Runtime → Run all**.
3. Review the file-integrity checks, prediction checks, and three official evaluator runs. The final cell downloads the model package and evaluation results.

The notebook embeds the model, weights, public demo, and unmodified official harness. An uploaded copy needs no GitHub sign-in or Google Drive mount. NumPy installs in a separate folder; installation is separate from model setup. The checks enforce setup under 10 seconds and all 400 predictions under 10 seconds. Run the notebook to measure performance on the assigned Colab CPU.

## Repository guide

| Path | Purpose |
| --- | --- |
| [`model.py`](model.py) | Competition entry point and tree inference |
| [`weights.npz`](weights.npz) | Trained tree parameters, feature indices, and class offsets |
| [`fast_numpy_features.py`](fast_numpy_features.py) | Selected preprocessing and vectorized feature extraction |
| [`core_features.py`](core_features.py), [`fused_features.py`](fused_features.py) | Earlier compiled feature implementations, retained for development history |
| [`requirements.txt`](requirements.txt) | Pinned inference dependencies |
| [`verify_colab.ipynb`](verify_colab.ipynb) | Self-contained verification notebook |
| [`training/`](training/) | Training, export, packaging, and verification utilities |
| [`APPROACH.md`](APPROACH.md), [`RESULTS.md`](RESULTS.md) | Method and experimental results |
| [`evidence/latest/`](evidence/latest/) | Recorded audits, comparisons, and runtime checks |
| [`VERIFICATION.json`](VERIFICATION.json) | Machine-readable verification summary |
| [`MANIFEST.json`](MANIFEST.json) | SHA-256 file hashes; excludes the manifest itself |

## Submission notes

The repository must be public or shared with the organizers' designated GitHub accounts so they can clone it. Organizer access has not been independently confirmed. Team and member details should accompany the submission through the organizers' required channel; no submission email has been sent by this project.

AI coding assistance was used in implementation, experimentation, and documentation. The method and its limitations are documented for review.
