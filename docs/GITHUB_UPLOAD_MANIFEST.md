# GitHub Upload Manifest

This document defines what should be uploaded to the STMAN GitHub repository and what must remain local.

## Repository Scope

The GitHub repository should contain the engineering code needed to reproduce STMAN training, baseline comparison, analysis scripts, and lightweight tests. It should not contain checkpoints, generated prediction files, raw public datasets, prepared data arrays, logs, figures, or paper response artifacts.

## Upload These Files

### STMAN core modules

Upload the top-level STMAN implementation files:

- `model.py`
- `train.py`
- `train_stman.py`
- `utils.py`
- `metric.py`
- `fractal_features.py`
- `batch_eval.py`
- `export_stman_predictions.py`
- `requirements.txt`
- `README.md`

These files define the STMAN architecture, fractal feature extraction, data loading, training, evaluation, and prediction export workflow.

### Baseline source code

Upload source-only baseline implementations under `baselines/`:

- `baselines/Graph-WaveNet/`
- `baselines/DMFGCRN/`
- `baselines/MTEGCRN/`
- `baselines/model/`
- `baselines/*.py`
- `baselines/*.sh`

The uploaded baseline folders should include Python modules, configuration files, shell launchers, README files, and license files where available. Baseline `data/`, `experiments/`, logs, checkpoints, generated predictions, and prepared `.npz/.npy/.pkl` files must not be uploaded.

### Experiment and analysis scripts

Upload reusable scripts under `scripts/`:

- `scripts/analysis/`
- `scripts/orchestration/`
- `scripts/revision_audit/`

These scripts cover complexity profiling, fractal-accuracy analysis, table/figure audit helpers, and multi-run orchestration. Runtime queue state files and generated monitor state directories should remain local.

### Dataset documentation

Upload dataset documentation only:

- `datasets/README.md`

Raw and processed datasets should remain local under ignored directories such as `datasets/raw/`, `datasets/processed/`, or `data/`.

### Tests

Upload lightweight tests under:

- `tests/`

The tests should verify model construction, checkpoint policy, data split policy, and training resume behavior without requiring large datasets or checkpoints.

## Do Not Upload

Do not upload any of the following:

- `checkpoint/`, `checkpoints/`
- `results/`, `runs/`, `outputs/`, `experiments/`
- `data/`, `Datasets/`, `datasets/raw/`, `datasets/processed/`
- generated fractal feature arrays such as `*.npy` or `*.npz`
- model weights such as `*.pt`, `*.pth`, `*.ckpt`
- prediction dumps, prepared baseline arrays, and serialized objects such as `*.pkl`, `*.h5`, `*.hdf5`
- logs such as `*.log`
- generated figures and paper artifacts such as `*.png`, `*.pdf`, `*.jpg`, `*.jpeg`
- notebooks such as `*.ipynb`
- Python caches and test caches

## Current Organized Layout

```text
STMAN/
  README.md
  requirements.txt
  model.py
  train.py
  train_stman.py
  utils.py
  metric.py
  fractal_features.py
  batch_eval.py
  export_stman_predictions.py
  baselines/
    Graph-WaveNet/
    DMFGCRN/
    MTEGCRN/
    model/
  scripts/
    analysis/
    orchestration/
    revision_audit/
  datasets/
    README.md
  tests/
  docs/
    GITHUB_UPLOAD_MANIFEST.md
```

## Git Workflow

Recommended commands:

```bash
git status --short
git add README.md .gitignore docs/GITHUB_UPLOAD_MANIFEST.md datasets/README.md baselines scripts tests *.py requirements.txt
git status --short
git commit -m "Organize STMAN code and baseline scripts"
git push origin main
```

Before committing, check that no large data or generated artifacts are staged:

```bash
git diff --cached --name-only
git diff --cached --stat
```

The repository is ready for GitHub only when the staged files contain source code, documentation, configs, and tests.
