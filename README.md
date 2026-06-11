# STMAN

This is the code repository for **Urban Wide-Area Traffic Flow Forecasting: A Spatiotemporal Multifractal Attention-Based Approach**.

## Repository Contents

- `model.py`: STMAN model definition.
- `train.py`: training loop, checkpoint handling, and evaluation helpers.
- `train_stman.py`: command-line training entry point.
- `utils.py`: data loading, preprocessing, calendar features, adjacency loading, and schedulers.
- `metric.py`: STMAN loss and metric utilities.
- `fractal_features.py`: temporal MF-DFA and spatial fractal feature extraction.
- `baselines/`: source-only baseline implementations and a clean baseline launcher.
- `scripts/`: reusable analysis and STMAN launch scripts.
- `Datasets/`: processed public traffic arrays and graph files used by the training scripts.
- `datasets/README.md`: dataset layout and file description.
- `requirements.txt`: Python dependencies.

Processed traffic arrays, adjacency matrices, and node-level fractal features are included for reproducibility. Checkpoints, logs, generated predictions, generated figures, raw `.h5` files, and local manuscript artifacts are not included.

## Quick Start

Install Git LFS and pull the processed data files:

```bash
git lfs install
git lfs pull
```

Run STMAN on one dataset:

```bash
python train_stman.py --dataset PEMS08 --seq_len 96 --pred_len 12 --checkpoint_dir checkpoints/stman
```

Run the public STMAN experiment launcher:

```bash
GPU=0 DATASETS="PEMS08 METR-LA" HORIZONS="12 48 96 288" bash scripts/run_stman_all.sh
```

Run a supported baseline:

```bash
GPU=0 bash baselines/run_baseline.sh dmfgcrn PEMS08 42
```
