# STMAN

This is the code repository for **Urban Wide-Area Traffic Flow Forecasting: A Spatiotemporal Multifractal Attention-Based Approach**.

## Repository Contents

- `model.py`: STMAN model definition.
- `train.py`: training loop, checkpoint handling, and evaluation helpers.
- `utils.py`: data loading, preprocessing, calendar features, adjacency loading, and schedulers.
- `metric.py`: STMAN loss and metric utilities.
- `fractal_features.py`: temporal MF-DFA and spatial fractal feature extraction.
- `baselines/`: source-only baseline implementations and a clean baseline launcher.
- `scripts/`:  STMAN launch scripts.
- `Datasets/`: processed public traffic arrays and graph files used by the training scripts.
- `FractalFeatures/`: node-level spatial and temporal fractal feature arrays.
- `datasets/README.md`: dataset layout and file description.
- `requirements.txt`: Python dependencies.

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
