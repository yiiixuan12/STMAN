#!/usr/bin/env python
"""Prepare Graph WaveNet data/adjs for revision baseline runs.

The original Graph WaveNet helper expects HDF inputs with a DatetimeIndex.
This revision package also stores PEMS03/PEMS07/PEMS08 as npz/csv arrays, so
this script builds the same train/val/test npz layout from those arrays and
writes Graph WaveNet-compatible adjacency pickles.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = {
    "METR-LA": {
        "nodes": 207,
        "start": "2012-03-01",
        "source": "Datasets/METR-LA/metr-la.h5",
        "adj": "Datasets/METR-LA/adj_matrix.csv",
    },
    "PEMS-BAY": {
        "nodes": 325,
        "start": "2017-01-01",
        "source": "Datasets/PEMS-BAY/pems-bay.h5",
        "adj": "Datasets/PEMS-BAY/adj_matrix.csv",
    },
    "PEMS03": {
        "nodes": 358,
        "start": "2018-09-01",
        "source": "Datasets/PEMS03/PEMS03.npz",
        "adj": "Datasets/PEMS03/adj_matrix.csv",
    },
    "PEMS07": {
        "nodes": 883,
        "start": "2017-05-01",
        "source": "Datasets/PEMS07/PEMS07.npz",
        "adj": "Datasets/PEMS07/adj_matrix.csv",
    },
    "PEMS08": {
        "nodes": 170,
        "start": "2016-07-01",
        "source": "Datasets/PEMS08/PEMS08.npz",
        "adj": "Datasets/PEMS08/adj_matrix.csv",
    },
}


def load_series(package_code_root: Path, dataset: str) -> pd.DataFrame:
    spec = DATASETS[dataset]
    source = package_code_root / spec["source"]
    if source.suffix == ".h5":
        try:
            frame = pd.read_hdf(source)
            return frame.astype(np.float32)
        except ImportError:
            csv_source = package_code_root / "Datasets" / dataset / f"{dataset}.csv"
            if not csv_source.exists():
                raise
            values = pd.read_csv(csv_source, header=None).values.astype(np.float32)
            index = pd.date_range(spec["start"], periods=values.shape[0], freq="5min")
            columns = [str(i) for i in range(values.shape[1])]
            return pd.DataFrame(values, index=index, columns=columns)

    if source.suffix == ".npz":
        raw = np.load(source)["data"]
        values = raw[..., 0] if raw.ndim == 3 else raw
    else:
        values = pd.read_csv(source).values

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D data for {dataset}, got {values.shape}")
    index = pd.date_range(spec["start"], periods=values.shape[0], freq="5min")
    columns = [str(i) for i in range(values.shape[1])]
    return pd.DataFrame(values, index=index, columns=columns)


def generate_graph_seq2seq_io_data(
    frame: pd.DataFrame,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    add_time_in_day: bool = True,
    add_day_in_week: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    num_samples, num_nodes = frame.shape
    data = np.expand_dims(frame.values, axis=-1)
    feature_list = [data]
    if add_time_in_day:
        time_ind = (
            frame.index.values - frame.index.values.astype("datetime64[D]")
        ) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day.astype(np.float32))
    if add_day_in_week:
        dow = frame.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled.astype(np.float32))

    full = np.concatenate(feature_list, axis=-1).astype(np.float32)
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    num_windows = max_t - min_t
    x = np.empty((num_windows, len(x_offsets), num_nodes, full.shape[-1]), dtype=np.float32)
    y = np.empty((num_windows, len(y_offsets), num_nodes, full.shape[-1]), dtype=np.float32)
    for index, t in enumerate(range(min_t, max_t)):
        x[index] = full[t + x_offsets, ...]
        y[index] = full[t + y_offsets, ...]
    return x, y


def split_files_valid(output_dir: Path) -> bool:
    for name in ("train", "val", "test"):
        path = output_dir / f"{name}.npz"
        if not path.exists():
            return False
        try:
            with np.load(path) as data:
                for key in ("x", "y", "x_offsets", "y_offsets"):
                    if key not in data:
                        return False
                if data["x"].shape[0] == 0 or data["y"].shape[0] == 0:
                    return False
        except Exception:
            return False
    return True


def write_splits(
    output_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    overwrite: bool,
) -> None:
    if output_dir.exists():
        if not overwrite and split_files_valid(output_dir):
            return
        shutil.rmtree(output_dir)

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    num_samples = x.shape[0]
    num_train = round(num_samples * train_ratio)
    num_test = round(num_samples * test_ratio)
    num_val = num_samples - num_train - num_test
    splits = {
        "train": (x[:num_train], y[:num_train]),
        "val": (x[num_train : num_train + num_val], y[num_train : num_train + num_val]),
        "test": (x[-num_test:], y[-num_test:]),
    }
    tmp_dir = output_dir.parent / f".{output_dir.name}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    for name, (split_x, split_y) in splits.items():
        np.savez_compressed(
            tmp_dir / f"{name}.npz",
            x=split_x,
            y=split_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
    if output_dir.exists():
        if not overwrite and split_files_valid(output_dir):
            shutil.rmtree(tmp_dir)
            return
        shutil.rmtree(output_dir)
    tmp_dir.rename(output_dir)


def write_adj_pickle(package_code_root: Path, dataset: str, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return
    spec = DATASETS[dataset]
    adj_path = package_code_root / spec["adj"]
    adj = np.loadtxt(adj_path, delimiter=",", dtype=np.float32)
    if adj.shape != (spec["nodes"], spec["nodes"]):
        raise ValueError(f"Unexpected adjacency shape for {dataset}: {adj.shape}")
    adj = (adj > 0).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    sensor_ids = [str(i) for i in range(spec["nodes"])]
    sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        pickle.dump((sensor_ids, sensor_id_to_ind, adj), fh, protocol=pickle.HIGHEST_PROTOCOL)


def prepare_dataset(args: argparse.Namespace, dataset: str) -> None:
    code_root = args.code_root.resolve()
    package_root = code_root.parents[1]
    dataset_key = dataset.lower().replace("-", "")
    output_dir = (
        args.output_root.resolve()
        / "graphwavenet"
        / dataset_key
        / f"h{args.horizon}_622"
    )

    print(f"{dataset}: preparing Graph WaveNet windows", flush=True)
    frame = load_series(code_root, dataset)
    x_offsets = np.sort(np.arange(-(args.seq_length_x - 1), 1, 1))
    y_offsets = np.sort(np.arange(args.y_start, args.horizon + 1, 1))
    x, y = generate_graph_seq2seq_io_data(
        frame,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=args.add_day_in_week,
    )
    write_splits(
        output_dir,
        x,
        y,
        x_offsets,
        y_offsets,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        overwrite=args.overwrite,
    )

    adj_name = f"adj_mx_{dataset_key}.pkl"
    if dataset == "METR-LA":
        adj_name = "adj_mx.pkl"
    elif dataset == "PEMS-BAY":
        adj_name = "adj_mx_bay.pkl"
    adj_out = code_root / "baseline" / "Graph-WaveNet" / "data" / "sensor_graph" / adj_name
    write_adj_pickle(code_root, dataset, adj_out, overwrite=args.overwrite_adj)

    print(
        f"{dataset}: data={output_dir} x={x.shape} y={y.shape} adj={adj_out}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "results"
        / "revision_baselines_12step_short"
        / "prepared_data",
    )
    parser.add_argument("--datasets", type=str, default="PEMS03,PEMS07,PEMS08")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seq_length_x", type=int, default=12)
    parser.add_argument("--y_start", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--add_day_in_week", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_adj", action="store_true")
    args = parser.parse_args()

    for dataset in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset {dataset}; choices={sorted(DATASETS)}")
        prepare_dataset(args, dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
