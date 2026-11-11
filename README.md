# GDG UAM competition model

This is the selected verified entry: **329/400 (82.25%)** on the public demo. The faster runtime retains the existing weights, which were refitted on all 100,000 official training examples.

## Run in Colab

Open [the verification notebook](https://colab.research.google.com/github/citraaparamithaaa-dev/gdg-uam/blob/main/verify_colab.ipynb), select a **CPU** runtime, connect, and choose **Run all**. You can also download [verify_colab.ipynb](verify_colab.ipynb) and upload it through Colab's File menu. The notebook embeds the exact model files, weights, public demo and unmodified official evaluator, so an uploaded copy needs no GitHub login or Google Drive mount.

Model packages install in a separate folder. Installation and initial compilation take longer than prediction. Moderate compiler optimization reduces cold setup cost. The notebook checks file hashes, prediction safety, the time for all 400 predictions, and three official evaluator runs, then downloads the model and results.

**Colab execution remains unverified here.** Run the notebook to measure that CPU. No winning rank or hidden-test accuracy has been established.

## Measured local results

| Check | Result |
|---|---:|
| Public demo | 329/400 (82.25%) |
| All 400 predictions, latest three checks | Median 0.694 seconds |
| Fresh official evaluator runs | Three passed |
| Exceptions, invalid outputs and timeouts | Zero |
| Maximum setup time in those fresh runs | 33.96 seconds (limit: 120) |
| Standalone notebook-payload check | Passed |

The demo has been repeatedly used during development, including earlier preprocessing selection. These are development results. The final-refit model has seen all training-file rows, so its accuracy on those rows is not independent validation. No hard accuracy ceiling has been proved.

The new CatBoost, blended, XGBoost and covariate-weighted candidates reached 79.25%, 82.00%, 81.00% and 81.50% on the demo, respectively. They did not replace the selected model. See [RESULTS.md](RESULTS.md), [APPROACH.md](APPROACH.md), and [evidence/latest](evidence/latest).

## Local evaluation

Use Python 3.12 and the pinned [requirements](requirements.txt). In the organizer's materials directory:

```text
python -m pip install -r PATH_TO_REPOSITORY/requirements.txt
python harness.py evaluate --model PATH_TO_REPOSITORY/model.py --source file:data/test_demo.npz --runs 3
```

`model.py` defines `Model`. Weights and helpers load relative to that file. Predictions return Python integers in 0..7. The runtime handles missing values, preserves read-only inputs, uses one CPU thread under the official evaluator, and uses no metadata, network access, GPU, training, label lookup, or background prediction during inference. Compilation occurs during setup. The organizer supplies `competition_api.py` and `harness.py`; they must be on the import path.

The original training workflow remains under `training/`. After exporting the balanced 231-feature model, [training/install_fast_runtime.py](training/install_fast_runtime.py) installs this optimized runtime around the numerical weights. Runtime checks and file hashes are in [VERIFICATION.json](VERIFICATION.json) and [MANIFEST.json](MANIFEST.json). The manifest excludes itself.

Before the competition deadline, ensure the organizers can clone the repository, run the Colab check, and send the required team details and repository link. No submission email has been sent from this project update. AI coding assistance was used; team members should be ready to explain the method.

Official rules and evaluator: https://github.com/GDG-UAM/ml-contest-materials

To regenerate the embedded notebook from these files and a local copy of the organizer's materials:

```text
python training/build_verify_notebook.py --materials PATH_TO_OFFICIAL_MATERIALS --candidate . --evidence evidence/latest/github_checkout_contract.json --output verify_colab.ipynb
```
