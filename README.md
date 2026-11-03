# GDG UAM time series classification entry

This repository contains a trained eight-class classifier for the GDG UAM machine learning contest. The selected development configuration is **balanced_core_200**. The submitted weights are refitted on all 100,000 official training rows after model selection. Demo labels are never used for fitting.

## Run the entry

Use Python 3.12 with the versions in `requirements.txt`. In the official materials folder:

For the evaluation-platform check, [open the CPU verification notebook in Google Colab](https://colab.research.google.com/github/citraaparamithaaa-dev/gdg-uam/blob/main/verify_colab.ipynb) and run its cells. The repository URL is already configured.

```powershell
python -m pip install -r PATH_TO_THIS_REPOSITORY/requirements.txt
python harness.py evaluate --model PATH_TO_THIS_REPOSITORY/model.py --source file:data/test_demo.npz
```

`model.py` defines `Model`. Initialization loads `weights.npz` and compiles CPU prediction routines. Each prediction returns a Python integer from 0 through 7. Missing values are interpolated; entirely missing channels become zeros. Inputs are copied before preprocessing. No label lookup, network request, GPU, training, or background prediction is used during inference. The official API and harness are supplied by the organizer.

## Approach

Each channel is centered and scaled. Features describe its distribution, local changes, autocorrelation, frequency content, and relationships between channels. The structural variant also measures frequency and amplitude changes across time and their behavior in a shared component of channels 0 and 1. The core variant retains the first two channels' shape and spectral features, six channel correlations, and the first two channels' and shared component's structural features. Absolute channel level, absolute channel scale, and metadata are excluded. A histogram gradient boosting classifier is trained on those summaries. Its numerical trees are exported to NumPy arrays and traversed with Numba on one CPU thread. Export parity is checked against the fitted scikit-learn model.

The balanced variant adds eight fixed class offsets to the model's scores. These offsets are calibrated to improve the competition's geometric mean objective using 5,000 rows from the training-file holdout; they never use demo labels. The remaining 15,000 holdout rows are excluded from both classifier fitting and offset calibration. The holdout was consulted during feature selection, so this remains development evidence. The offsets are retained when the classifier is refitted on all training rows.

## Development evidence

All candidates begin with the same stratified seed-42 split: 80,000 training rows and 20,000 held-out rows. Uncalibrated candidates are checked on all 20,000 holdout rows; the balanced candidate uses 5,000 for calibration and 15,000 for checking. The saved split indices are in `evidence/`. Development metrics belong to models fitted on the 80,000 rows, not to the final refit.

| Candidate | Check rows | Held-out accuracy | Demo diagnostic accuracy | Weakest demo class recall |
| --- | ---: | ---: | ---: | ---: |
| balanced_core_200 | 15,000 | 87.67% | 76.75% | 44.00% |
| baseline_300 | 20,000 | 82.41% | 40.25% | 0.00% |
| baseline_shape_300 | 20,000 | 71.98% | 50.50% | 6.00% |
| core_structure_200 | 20,000 | 87.96% | 75.25% | 26.00% |
| relations_150 | 20,000 | 87.05% | 67.75% | 12.00% |
| relations_300 | 20,000 | 87.39% | 69.00% | 16.00% |
| shape_150 | 20,000 | 86.20% | 68.75% | 22.00% |
| structure_150 | 20,000 | 88.51% | 72.50% | 22.00% |
| structure_300 | 20,000 | 89.00% | 73.75% | 20.00% |

A second training-only split (seed 123) gave 87.33% accuracy. It is a split-sensitivity check, not an untouched test set.

The 400-row demo contains only 50 examples per class. It was inspected as a distribution-shift diagnostic during development, so its results are not an untouched estimate of hidden-test performance. One changed prediction changes overall demo accuracy by 0.25 percentage points. The validation split follows the training distribution; it can overestimate performance after a distribution shift. Final refit harness results are recorded separately in `evidence/` when available. Finite-file scores cannot be interpreted as the score of an infinite 60-second official stream.

## Final refit harness check

On the supplied 400-row demo, the final refit achieved 77.25% accuracy across three runs, mean finite-file score 2.5650, mean prediction latency 2.118 ms, and 0 failed runs. These are local Windows CPU results, not the hidden competition score or a Colab timing result. The notebook `verify_colab.ipynb` is provided for the remaining Colab CPU check.



## Reproduce training

Keep the competition data out of this repository. The following commands obtain the original materials, add these training scripts, recreate the experiments, refit the selected design, and produce another inference package. Training requires the additional libraries in `training/requirements-training.txt`.

```powershell
git clone https://github.com/GDG-UAM/ml-contest-materials.git work
Copy-Item .\training\* .\work\
Set-Location work
python -m pip install -r requirements-training.txt
python download_data.py
python prepare_split.py
python train_candidates.py
python train_structure.py
python calibrate_bias.py core_structure_200 --out balanced_core_200
python audit_candidate.py balanced_core_200
python train_candidates.py --final balanced_core_200
python export_model.py experiments/balanced_core_200/final.pkl ../refit-package
```

Training settings:

```json
{
  "max_iter": 200,
  "max_leaf_nodes": 15,
  "learning_rate": 0.08,
  "l2_regularization": 1.0,
  "random_state": 42,
  "early_stopping": false,
  "max_bins": 128
}
```

Training uses four CPU threads for efficiency; the official harness limits inference to one. Seeds and fixed split indices make the workflow repeatable in the tested environment. Small numerical differences across platforms and library versions remain possible. Linux/Colab timing must be checked separately.

## Sources and limits

Official materials: https://github.com/GDG-UAM/ml-contest-materials at commit `fc4566d3e3f54fa31a0daa4e3f99a55d783b021a`.

Training data SHA256: `1c231e9a807c348b34290415f56e7c7b99543a81673a9d3cef1002b1a42e5668`.

Demo data SHA256: `03fce51b3dacfaaa04418c4907def59874c4b6963c0561c9ccac83bbca90aac0`.

This entry was developed with AI coding assistance. Team members should understand and be able to explain the feature extraction, validation limitations, and exported model. Team registration, member details, deadline timezone, repository access for organizers, and submission email are separate administrative steps.
