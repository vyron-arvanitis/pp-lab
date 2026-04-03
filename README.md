# Smart Background AI Lab

This repository now uses a standard Python project layout:

```text
src/pp_lab/
  cli/          command-line entry points
  config.py     shared paths and feature-column settings
  data.py       dataset loading, preprocessing, padding, batching
  models.py     model definitions and registry
  training.py   fit/loss/accuracy helpers
  reporting.py  analysis and comparison helpers
scripts/        direct script wrappers
```

The generated experiment outputs, paper assets, and notebooks are still kept in their existing folders, but the reusable Python code no longer lives at the repository root.

## Run

Use either the packaged entry points after installation or the direct wrappers:

```powershell
pip install -e .
pp-train --help
pp-compare-models --help
```

Or without installation:

```powershell
python scripts/train_workbook.py --help
python scripts/compare_model_features.py --help
```

The old top-level files such as `workbook.py`, `models.py`, and `utils.py` are now thin compatibility wrappers.

## Existing Project Assets

- `models_fulltrain/`, `saved_models/`, `optuna_results/`: training outputs
- `AI Lab 3 - PP/`, `MD_Files/`, `figures/`: report and documentation material
- `dataset_and_models.ipynb`, `loading_and_evaluation.ipynb`: notebooks
