# STMAN

This is the code repository for **Urban Wide-Area Traffic Flow Forecasting: A Spatiotemporal Multifractal Attention-Based Approach**.

## Repository Contents

- `model.py`: STMAN model definition.
- `train.py`: training loop, checkpoint handling, and evaluation helpers.
- `train_stman.py`: command-line training entry point.
- `utils.py`: data loading, preprocessing, calendar features, adjacency loading, and schedulers.
- `metric.py`: STMAN loss and metric utilities.
- `fractal_features.py`: temporal MF-DFA and spatial fractal feature extraction.
- `batch_eval.py`: batch evaluation utilities.
- `export_stman_predictions.py`: prediction export utility.
- `baselines/`: source-only baseline implementations and launch scripts.
- `scripts/`: analysis, orchestration, and revision-audit utilities.
- `datasets/README.md`: local dataset layout instructions.
- `docs/GITHUB_UPLOAD_MANIFEST.md`: upload checklist and ignored-artifact policy.
- `tests/`: lightweight code checks for core model and training policies.
- `requirements.txt`: Python dependencies.

Datasets, checkpoints, logs, generated features, and experimental results are not included.

## GitHub Upload Policy

Only source code, configuration files, tests, and documentation should be committed. Raw datasets, prepared arrays, checkpoints, prediction dumps, logs, generated figures, and paper-response artifacts should remain local. See `docs/GITHUB_UPLOAD_MANIFEST.md` for the full upload checklist.
