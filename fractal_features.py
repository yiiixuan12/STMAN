#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from numpy.polynomial.polynomial import polyfit, polyval


def compute_profile(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    return np.cumsum(series - np.mean(series))


def windowing(series: np.ndarray, scale: int) -> np.ndarray:
    if not series.flags["C_CONTIGUOUS"]:
        series = series.copy()
    return as_strided(
        series,
        (series.shape[0] // scale, scale),
        ((scale * series.dtype.itemsize), series.dtype.itemsize),
    )


def bidirectional_mean_variance(profile: np.ndarray, scales: np.ndarray, order: int = 1) -> np.ndarray:
    out = np.empty((len(scales), 2 * profile.shape[0] // scales[0]), dtype=np.float64)
    out[:] = np.nan
    inverse = profile[::-1]
    for index, scale in enumerate(scales):
        offset = int((out.shape[1] // 2) - (profile.shape[0] // scale))
        windows = np.concatenate((windowing(profile, scale), windowing(inverse, scale)))
        t = np.arange(scale)
        coef = polyfit(t, windows.T, order)
        var = np.mean((windows - polyval(t, coef)) ** 2, axis=1)
        out[index, offset : offset + var.shape[0]] = var
    return out


def fluctuation_function(mean_variance: np.ndarray, qs: np.ndarray) -> np.ndarray:
    out = np.zeros((mean_variance.shape[0], len(qs)), dtype=np.float64)
    mv = np.nan_to_num(mean_variance)
    masked = np.ma.array(mv, mask=mv < 5e-4, dtype=np.float64)
    for q_index, q_value in enumerate(qs):
        if q_value == 0:
            continue
        out[:, q_index] = (masked ** (q_value / 2)).mean(1) ** (1.0 / q_value)
    out[:, qs == 0] = np.exp(0.5 * np.ma.log(masked).mean(1))[:, None]
    return out


def get_scales(length: int) -> np.ndarray:
    scales = []
    n = 6
    while 2 ** np.floor(n / 2.0) + 2 ** np.floor((n + 1.0) / 2.0) < length / 3:
        scales.append(2 ** np.floor(n / 2.0) + 2 ** np.floor((n + 1.0) / 2.0))
        n += 1
    return np.asarray(scales, dtype=np.int32)


def mfdfa(series: np.ndarray, polynomial_order: int = 1) -> dict[str, np.ndarray | float]:
    scales = get_scales(series.size)
    qs = np.arange(-5.0, 5.01, 0.25)
    if len(scales) < 3:
        raise ValueError("time series is too short for MF-DFA")

    profile = compute_profile(series)
    mean_variance = bidirectional_mean_variance(profile, scales, order=polynomial_order)
    fq = fluctuation_function(mean_variance, qs)
    hq = np.zeros(len(qs), dtype=np.float64)
    coeff = np.zeros((len(qs), 2), dtype=np.float64)

    for q_index, _q_value in enumerate(qs):
        mask = fq[:, q_index] == 0
        valid = ~mask
        if valid.sum() < 3:
            hq[q_index] = 0.5
            continue
        coefs = np.ma.polyfit(
            np.ma.log2(np.ma.array(scales, mask=mask)),
            np.ma.log2(np.ma.array(fq[:, q_index], mask=mask)),
            1,
        )
        coeff[q_index] = coefs
        hq[q_index] = coefs[0]

    tq = hq * qs - 1
    alpha = np.diff(tq) / (qs[1] - qs[0])
    spectrum = (qs[:-1] * alpha) - tq[:-1]
    width = float(np.nanmax(alpha) - np.nanmin(alpha))
    h0 = float(hq[np.where(np.isclose(qs, 0.0))[0][0]])
    asymmetry = 0.0 if width == 0 else float(((np.nanmax(alpha) - h0) - (h0 - np.nanmin(alpha))) / width)
    return {
        "qs": qs,
        "scales": scales,
        "scaling_function": fq,
        "h_q": hq,
        "polynomial_coeff": coeff,
        "alpha": alpha,
        "spectrum": spectrum,
        "width": width,
        "asymmetry": asymmetry,
    }


def temporal_fractal_vector(series: np.ndarray) -> np.ndarray:
    result = mfdfa(np.asarray(series, dtype=np.float64))
    qs = result["qs"]
    hq = result["h_q"]
    selected = [float(hq[np.where(np.isclose(qs, q))[0][0]]) for q in (-4.0, 0.0, 2.0, 4.0, 5.0)]
    return np.asarray(selected + [float(result["width"]), float(result["asymmetry"])], dtype=np.float32)


def shortest_path_distances(adj_matrix: np.ndarray) -> np.ndarray:
    dist = np.asarray(adj_matrix, dtype=np.float64).copy()
    dist[dist == 0] = np.inf
    np.fill_diagonal(dist, 0.0)
    dist = np.minimum(dist, dist.T)
    for k in range(dist.shape[0]):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist


def spatial_fractal_vectors(adj_matrix: np.ndarray, min_scale: float = 1.0, max_scale: float = 8.0, num_scales: int = 8) -> np.ndarray:
    distances = shortest_path_distances(adj_matrix)
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
    log_scales = np.log10(scales)
    rows = []
    for node_index in range(distances.shape[0]):
        counts = np.asarray([np.sum(distances[node_index] <= scale) for scale in scales], dtype=np.float64)
        log_counts = np.log10(np.clip(counts, 1.0, None))
        valid = np.isfinite(log_counts)
        if valid.sum() >= 3:
            slope, intercept = np.polyfit(log_scales[valid], log_counts[valid], 1)
        else:
            slope, intercept = 0.0, 0.0
        rows.append(np.concatenate([log_counts, [slope, intercept]]))
    return np.asarray(rows, dtype=np.float32)


def load_traffic_array(dataset_dir: Path, dataset: str) -> np.ndarray:
    npz_path = dataset_dir / f"{dataset}.npz"
    csv_path = dataset_dir / f"{dataset}.csv"
    h5_path = dataset_dir / f"{dataset.lower()}.h5"
    if npz_path.exists():
        data = np.load(npz_path)
        array = data["data"]
    elif csv_path.exists():
        array = pd.read_csv(csv_path).values.astype(np.float32)
        array = array.reshape(array.shape[0], array.shape[1], 1)
    elif h5_path.exists():
        array = pd.read_hdf(h5_path).values.astype(np.float32)
        array = array.reshape(array.shape[0], array.shape[1], 1)
    else:
        raise FileNotFoundError(f"cannot find traffic data under {dataset_dir}")
    if array.ndim == 2:
        array = array[..., None]
    if array.shape[-1] > 1:
        array = array[..., :1]
    return array.astype(np.float32)


def load_adjacency(dataset_dir: Path) -> np.ndarray:
    path = dataset_dir / "adj_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"cannot find adjacency matrix: {path}")
    return pd.read_csv(path, header=None).values.astype(np.float32)


def extract_dataset_features(data_root: Path, dataset: str, output_dir: Path, train_ratio: float = 0.6) -> tuple[Path, Path]:
    dataset_dir = data_root / dataset
    traffic = load_traffic_array(dataset_dir, dataset)
    train_len = int(traffic.shape[0] * train_ratio)
    traffic_train = traffic[:train_len, :, 0]

    temporal = np.vstack([temporal_fractal_vector(traffic_train[:, i]) for i in range(traffic_train.shape[1])])
    spatial = spatial_fractal_vectors(load_adjacency(dataset_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    temporal_path = output_dir / f"train_timefractals_{dataset}.npy"
    spatial_path = output_dir / f"spatial_fractal_vectors_{dataset}.npy"
    np.save(temporal_path, temporal)
    np.save(spatial_path, spatial)
    return spatial_path, temporal_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("Datasets"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("."))
    parser.add_argument("--train_ratio", type=float, default=0.6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spatial_path, temporal_path = extract_dataset_features(
        data_root=args.data_root,
        dataset=args.dataset,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
    )
    print(f"wrote {spatial_path}")
    print(f"wrote {temporal_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
