#!/usr/bin/env python3
"""
Compute R^2 goodness-of-fit for multifractal analysis in STMAN.

Temporal fractals: MF-DFA log-log regression R^2 at q=2 (standard Hurst exponent)
Spatial fractals: Box-counting log-log regression R^2 (log(M(l)) vs log(l))

Reproduces the exact procedures from mutifract.ipynb.
"""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from numpy.polynomial.polynomial import polyval, polyfit

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent


def _r2_score(y_true, y_pred):
    """Small local replacement for sklearn.metrics.r2_score."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) < 2:
        return np.nan
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return 1.0 if ss_res <= 0 else 0.0
    return 1.0 - ss_res / ss_tot

# ============================================================
# MF-DFA code (reproduced from mutifract.ipynb)
# ============================================================

def compute_profile(X):
    return np.cumsum(X - np.mean(X))

def windowing(X, s):
    if not X.flags['C_CONTIGUOUS']:
        X = X.copy()
    return as_strided(X, (X.shape[0] // s, s), ((s * X.dtype.itemsize), X.dtype.itemsize))

def bidir_mean_variance(X, scales, m=1):
    out = np.empty((len(scales), 2 * X.shape[0] // scales[0]))
    out[:] = np.nan
    X_inverse = X[::-1]
    for j, scale in enumerate(scales):
        idx = int((out.shape[1] // 2) - (X.shape[0] // scale))
        Y = np.concatenate((windowing(X, scale), windowing(X_inverse, scale)))
        i = np.arange(scale)
        coef = polyfit(i, Y.T, m)
        var = np.mean((Y - polyval(i, coef)) ** 2, axis=1)
        out[j, idx:idx + var.shape[0]] = var
    return out

def compute_fluctuation_function(mv, qs):
    out = np.zeros((mv.shape[0], len(qs)), 'f8')
    mv = np.nan_to_num(mv)
    mask = mv < 0.0005
    maskedMV = np.ma.array(mv, mask=mask, dtype=np.float64)
    for qi, q in enumerate(qs):
        if q == 0:
            continue
        out[:, qi] = (maskedMV ** (q / 2)).mean(1) ** (1.0 / q)
    out[:, qs == 0] = np.exp(0.5 * (np.ma.log(maskedMV)).mean(1))[:, None]
    return out

def get_scales(length):
    scales = []
    n = 6
    while 2 ** np.floor(n / 2.) + 2 ** np.floor((n + 1.) / 2.) < length / 3:
        scales.append(2 ** np.floor(n / 2.) + 2 ** np.floor((n + 1.) / 2.))
        n += 1
    return np.array(scales).astype('i4')

def mfdfa_r2_at_q2(series):
    """
    Run MF-DFA on a single time series and return R^2 of the
    log(F(s)) vs log(s) regression at q=2 (standard Hurst exponent).
    """
    qs = np.arange(-5.0, 5.01, 0.25)
    scales = get_scales(len(series))

    if len(scales) < 3:
        return np.nan

    profile = compute_profile(series)
    bmv = bidir_mean_variance(profile, scales, m=1)
    Fq = compute_fluctuation_function(bmv, qs)

    # Find q=2 index
    q2_idx = np.where(np.abs(qs - 2.0) < 1e-6)[0]
    if len(q2_idx) == 0:
        return np.nan
    q2_idx = q2_idx[0]

    Fq_q2 = Fq[:, q2_idx]

    # Mask zeros
    mask = Fq_q2 == 0
    mScales = np.ma.array(scales, mask=mask)
    mFq = np.ma.array(Fq_q2, mask=mask)

    valid = ~mask
    if np.sum(valid) < 3:
        return np.nan

    try:
        coefs = np.ma.polyfit(np.ma.log2(mScales), np.ma.log2(mFq), 1)
        log_fq_actual = np.log2(mFq[~mask].data)
        log_fq_pred = np.polyval(coefs, np.log2(mScales[~mask].data))
        r2 = _r2_score(log_fq_actual, log_fq_pred)
        return r2
    except:
        return np.nan


# ============================================================
# Spatial fractal code (reproduced from mutifract.ipynb)
# ============================================================

def compute_spatial_r2(adj_matrix, min_scale=1, max_scale=8, num_scales=8):
    """
    Compute R^2 of log(M(l)) vs log(l) for each node using box-counting
    on the adjacency/distance graph, matching the notebook procedure exactly.

    Returns array of R^2 values, one per node.
    """
    n_nodes = adj_matrix.shape[0]

    # Compute shortest-path distance matrix. Keep this self-contained because the
    # Some ROCm training environments do not install scipy.
    dist_adj = adj_matrix.astype(float)
    dist_adj[dist_adj == 0] = np.inf
    np.fill_diagonal(dist_adj, 0)
    distance_matrix = np.minimum(dist_adj, dist_adj.T)
    for k in range(n_nodes):
        distance_matrix = np.minimum(
            distance_matrix,
            distance_matrix[:, [k]] + distance_matrix[[k], :],
        )

    # Scales (matching the notebook: logspace from min_scale to max_scale)
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)

    r2_values = np.empty(n_nodes)

    for node_idx in range(n_nodes):
        # Neighborhood expansion curve
        expansion_curve = np.array([
            np.sum(distance_matrix[node_idx] <= scale)
            for scale in scales
        ])

        log_scales = np.log10(scales)
        log_expansion = np.log10(expansion_curve)

        valid_mask = np.isfinite(log_expansion) & (expansion_curve > 0)

        if np.sum(valid_mask) < 3:
            r2_values[node_idx] = np.nan
            continue

        x = log_scales[valid_mask]
        y = log_expansion[valid_mask]

        coef = np.polyfit(x, y, 1)
        y_pred = np.polyval(coef, x)
        r2 = _r2_score(y, y_pred)
        r2_values[node_idx] = r2

    return r2_values


# ============================================================
# Dataset loading helpers
# ============================================================

def load_traffic_data(dataset_name):
    """Load traffic data for a given dataset, return shape (T, N, F)."""
    base = ROOT / "Datasets" / dataset_name

    if dataset_name in ['PEMS03', 'PEMS07', 'PEMS08']:
        data = np.load(base / f"{dataset_name}.npz")
        X = data['data']
        # For PEMS08 which has 3 features, take only the first (flow)
        if X.shape[-1] > 1:
            X = X[..., :1]
        return X
    elif dataset_name == 'METR-LA':
        df = pd.read_csv(base / "METR-LA.csv")
        X = df.values.astype(np.float32)
        return X.reshape(X.shape[0], X.shape[1], 1)
    elif dataset_name == 'PEMS-BAY':
        df = pd.read_csv(base / "PEMS-BAY.csv")
        X = df.values.astype(np.float32)
        return X.reshape(X.shape[0], X.shape[1], 1)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_adj_matrix(dataset_name):
    """Load adjacency matrix for a given dataset."""
    base = ROOT / "Datasets" / dataset_name
    adj = pd.read_csv(base / "adj_matrix.csv", header=None).values.astype(np.float32)
    return adj


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="code/multifract root containing Datasets/",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "r2_goodness",
        help="directory for CSV/manifest outputs",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="METR-LA,PEMS03,PEMS07,PEMS08,PEMS-BAY",
        help="comma-separated datasets",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ROOT = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]

    results = []
    node_rows = []

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {ds}")
        print(f"{'='*60}")

        # --- Temporal R^2 (MF-DFA at q=2) ---
        print(f"  Loading traffic data for {ds}...")
        X = load_traffic_data(ds)
        T = X.shape[0]
        N = X.shape[1]
        n_train = int(T * 0.6)
        X_train = X[:n_train]

        print(f"  X shape: {X.shape}, train: {X_train.shape}, N_nodes: {N}")
        print(f"  Computing temporal MF-DFA R^2 (q=2) for {N} nodes...")

        temporal_r2 = np.empty(N)
        for i in range(N):
            series = X_train[:, i, 0]
            temporal_r2[i] = mfdfa_r2_at_q2(series)
            if (i + 1) % 100 == 0 or i == N - 1:
                print(f"    Node {i+1}/{N} done")

        valid_temporal = temporal_r2[~np.isnan(temporal_r2)]
        print(f"  Temporal R^2: mean={np.mean(valid_temporal):.6f}, "
              f"std={np.std(valid_temporal):.6f}, "
              f"min={np.min(valid_temporal):.6f}, "
              f"max={np.max(valid_temporal):.6f}, "
              f"valid={len(valid_temporal)}/{N}")

        # --- Spatial R^2 (box-counting) ---
        print(f"  Loading adjacency matrix for {ds}...")
        adj = load_adj_matrix(ds)
        print(f"  Adj matrix shape: {adj.shape}")
        print(f"  Computing spatial box-counting R^2 for {N} nodes...")

        spatial_r2 = compute_spatial_r2(adj, min_scale=1, max_scale=8, num_scales=8)

        valid_spatial = spatial_r2[~np.isnan(spatial_r2)]
        print(f"  Spatial R^2: mean={np.mean(valid_spatial):.6f}, "
              f"std={np.std(valid_spatial):.6f}, "
              f"min={np.min(valid_spatial):.6f}, "
              f"max={np.max(valid_spatial):.6f}, "
              f"valid={len(valid_spatial)}/{N}")

        results.append({
            'dataset': ds,
            'n_nodes': N,
            'temporal_r2_mean': np.mean(valid_temporal),
            'temporal_r2_std': np.std(valid_temporal),
            'temporal_r2_min': np.min(valid_temporal),
            'temporal_r2_max': np.max(valid_temporal),
            'temporal_valid': len(valid_temporal),
            'spatial_r2_mean': np.mean(valid_spatial),
            'spatial_r2_std': np.std(valid_spatial),
            'spatial_r2_min': np.min(valid_spatial),
            'spatial_r2_max': np.max(valid_spatial),
            'spatial_valid': len(valid_spatial),
        })
        for node_idx in range(N):
            node_rows.append({
                "dataset": ds,
                "node_index": node_idx,
                "temporal_r2": temporal_r2[node_idx],
                "spatial_r2": spatial_r2[node_idx],
            })

    summary_csv = output_dir / "r2_goodness_summary.csv"
    nodes_csv = output_dir / "r2_goodness_nodes.csv"
    manifest_json = output_dir / "r2_goodness_manifest.json"
    pd.DataFrame(results).to_csv(summary_csv, index=False)
    pd.DataFrame(node_rows).to_csv(nodes_csv, index=False)
    manifest_json.write_text(
        json.dumps(
            {
                "root": str(ROOT),
                "datasets": datasets,
                "summary_csv": str(summary_csv),
                "nodes_csv": str(nodes_csv),
                "method": "temporal MF-DFA log-log R2 at q=2; spatial box-counting log-log R2",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Summary table
    print(f"\n\n{'='*100}")
    print("SUMMARY: R^2 Goodness-of-Fit for Multifractal Analysis")
    print(f"{'='*100}")
    print(f"{'Dataset':<12} {'N':>5} "
          f"{'Temporal R^2 (mean +/- std)':>30} {'Temp Min':>10} "
          f"{'Spatial R^2 (mean +/- std)':>30} {'Spat Min':>10} "
          f"{'Overall Min':>12}")
    print(f"{'-'*100}")

    for r in results:
        overall_min = min(r['temporal_r2_min'], r['spatial_r2_min'])
        print(f"{r['dataset']:<12} {r['n_nodes']:>5} "
              f"{r['temporal_r2_mean']:>12.6f} +/- {r['temporal_r2_std']:<12.6f} "
              f"{r['temporal_r2_min']:>10.6f} "
              f"{r['spatial_r2_mean']:>12.6f} +/- {r['spatial_r2_std']:<12.6f} "
              f"{r['spatial_r2_min']:>10.6f} "
              f"{overall_min:>12.6f}")
    print(f"{'='*100}")

    # Also print in the exact requested format
    print(f"\n\nLaTeX table:")
    print(f"{'Dataset':<12} | {'Temporal R^2 (mean +/- std)':^30} | {'Spatial R^2 (mean +/- std)':^30} | {'Min R^2':^12}")
    print(f"{'-'*12}-+-{'-'*30}-+-{'-'*30}-+-{'-'*12}")
    for r in results:
        overall_min = min(r['temporal_r2_min'], r['spatial_r2_min'])
        t_str = f"{r['temporal_r2_mean']:.4f} +/- {r['temporal_r2_std']:.4f}"
        s_str = f"{r['spatial_r2_mean']:.4f} +/- {r['spatial_r2_std']:.4f}"
        print(f"{r['dataset']:<12} | {t_str:^30} | {s_str:^30} | {overall_min:^12.4f}")
    print(f"\nWrote CSV outputs:\n  {summary_csv}\n  {nodes_csv}\n  {manifest_json}")
