# Dataset Organization

This repository includes the processed files needed to reproduce the experiments. Large arrays are managed by Git LFS; after cloning, run `git lfs pull` to download the actual data files.

- traffic time-series arrays: `Datasets/<dataset>/<dataset>.npz`
- adjacency matrices: `Datasets/<dataset>/adj_matrix.csv`
- optional edge lists, node maps, and sensor metadata where available
- node-level fractal features: `spatial_fractal_vectors_<dataset>.npy` and `train_timefractals_<dataset>.npy`

Large raw files such as `.h5`, generated train/validation/test splits, checkpoints, prediction dumps, logs, and figures are intentionally excluded.

Included dataset names:

- `PEMS03`
- `PEMS07`
- `PEMS08`
- `PEMS-BAY`
- `METR-LA`

```text
Datasets/
  PEMS03/
    PEMS03.npz
    adj_matrix.csv
    PEMS03_edges.csv
    node_index_map.csv
  PEMS07/
  PEMS08/
  PEMS-BAY/
  METR-LA/
spatial_fractal_vectors_<dataset>.npy
train_timefractals_<dataset>.npy
```

The training scripts read this layout directly through the default `Datasets/<dataset>/` path.
