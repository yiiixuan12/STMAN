# STMAN

This is the code repository for **Urban Wide-Area Traffic Flow Forecasting: A Spatiotemporal Multifractal Attention-Based Approach**.

## Repository Contents

- `model.py`: STMAN model definition.
- `train.py`: training loop, checkpoint handling, and evaluation helpers.
- `utils.py`: data loading, preprocessing, calendar features, adjacency loading, and schedulers.
- `metric.py`: STMAN loss and metric utilities.
- `fractal_features.py`: temporal MF-DFA and spatial fractal feature extraction.
- `baselines/`: source-only baseline implementations and a clean baseline launcher.
- `scripts/`: STMAN launcher helper; see the training entry-point note below.
- `Datasets/`: processed public traffic arrays and graph files used by the training scripts.
- `requirements.txt`: Python dependencies.

## Setup

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install Git LFS and pull the processed data files:

```bash
git lfs install
git lfs pull
```

Processed traffic arrays and graph files are organized under `Datasets/<dataset>/`.

## Fractal Feature Extraction

Precomputed fractal feature arrays are not included in the repository. Generate them locally with `fractal_features.py`:

```bash
python fractal_features.py --data_root Datasets --dataset PEMS08 --output_dir prepared_data/fractal_features
```

Replace `PEMS08` with `PEMS03`, `PEMS07`, `PEMS-BAY`, or `METR-LA` as needed. The extractor writes:

- `spatial_fractal_vectors_<dataset>.npy`: node-level spatial descriptors.
- `train_timefractals_<dataset>.npy`: temporal descriptors computed from the training partition only.

By default, the training partition is 60% of the series for PEMS03/07/08 and 70% for PEMS-BAY and METR-LA. Use `--train_ratio` to override this setting. Generated feature arrays remain local and are ignored by Git.

## STMAN Training APIs

`model.py` defines `STFractalTransformer`, and `train.py` provides `train_model`, evaluation helpers, and checkpoint handling. Use these APIs from a Python training driver with the data loaders, adjacency matrix, and locally generated fractal descriptors.

The current repository does not include a `train_stman.py` command-line driver. The helper `scripts/run_stman_all.sh` depends on that driver and requires it to be supplied before use.

## Baselines

Run a supported baseline:

```bash
GPU=0 bash baselines/run_baseline.sh dmfgcrn PEMS08 42
```
