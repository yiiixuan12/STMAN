# STMAN

This is the code repository for **Urban Wide-Area Traffic Flow Forecasting: A Spatiotemporal Multifractal Attention-Based Approach**.

## Contents

- `model.py`: STMAN model.
- `train.py`: Training and evaluation.
- `utils.py`: Data loading and preprocessing.
- `metric.py`: Loss functions and evaluation metrics.
- `fractal_features.py`: Spatial and temporal fractal feature extraction.
- `baselines/`: Baseline models.
- `Datasets/`: Traffic datasets and graph files.
- `scripts/`: Experiment scripts.
- `requirements.txt`: Python dependencies.

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download dataset files:

```bash
git lfs install
git lfs pull
```
