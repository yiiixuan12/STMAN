# Dataset Organization

The public datasets are not committed to this repository because of their size and redistribution constraints.

Place local data under one of the ignored directories below before running experiments:

- `datasets/raw/`
- `datasets/processed/`
- `data/`

Expected dataset names:

- `PEMS03`
- `PEMS07`
- `PEMS08`
- `PEMS-BAY`
- `METR-LA`

Recommended local layout:

```text
datasets/
  raw/
    PEMS03/
    PEMS07/
    PEMS08/
    PEMS-BAY/
    METR-LA/
  processed/
    PEMS03/
    PEMS07/
    PEMS08/
    PEMS-BAY/
    METR-LA/
```

Typical files used by the training scripts include traffic matrices, adjacency matrices, edge lists, and node index maps. Generated fractal features, prepared baseline data, model checkpoints, logs, and prediction outputs should remain outside Git and are ignored by `.gitignore`.
