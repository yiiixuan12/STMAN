#!/usr/bin/env python
"""Retrain LSTNN/PDFormer Table III cells with auditable 6:2:2 testing.

This runner is intentionally separate from the legacy Excel/table exporters.
Each cell trains from the local dataset files, evaluates the returned best
validation model on the test split, and writes machine-readable artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("lstnn", "pdformer")
DATASETS = ("METR-LA", "PEMS-BAY", "PEMS03", "PEMS07", "PEMS08")
HORIZONS = (12, 48, 96, 288, 864, 2016)
UNSUPPORTED = {("lstnn", "PEMS08", 2016), ("pdformer", "PEMS08", 2016)}


@dataclass(frozen=True)
class JobSpec:
    model: str
    dataset: str
    horizon: int
    seq_len: int
    gpu: int
    epochs: int
    batch_size: int
    seed: int
    run_dir: Path


def package_root_from_code_root(root: Path) -> Path:
    root = root.resolve()
    if root.name != "multifract":
        raise ValueError(f"Expected code root ending in multifract, got {root}")
    return root.parents[1]


def dataset_key(dataset: str) -> str:
    return dataset.lower().replace("-", "").replace("_", "").replace(" ", "")


def parse_csv_list(text: str, default: tuple) -> list[str]:
    if not text:
        return list(default)
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_int_list(text: str, default: tuple[int, ...]) -> list[int]:
    if not text:
        return list(default)
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def normalize_model(model: str) -> str:
    model = model.strip().lower()
    aliases = {"lsttn": "lstnn", "lstnn": "lstnn", "pdformer": "pdformer"}
    if model not in aliases:
        raise ValueError(f"Unknown model={model}; choices={MODELS}")
    return aliases[model]


def normalize_dataset(dataset: str) -> str:
    lookup = {dataset_key(item): item for item in DATASETS}
    key = dataset_key(dataset)
    if key not in lookup:
        raise ValueError(f"Unknown dataset={dataset}; choices={DATASETS}")
    return lookup[key]


def default_batch_size(model: str, dataset: str, horizon: int) -> int:
    if dataset == "PEMS07":
        if model == "lstnn":
            return 4 if horizon >= 288 else 8
        return 1 if model == "pdformer" or horizon >= 288 else 2
    if dataset == "PEMS03":
        if model == "lstnn":
            return 8 if horizon >= 288 else 16
        return 4 if model == "pdformer" or horizon >= 288 else 8
    if dataset == "PEMS-BAY":
        return 8 if model == "pdformer" or horizon >= 288 else 16
    if dataset == "METR-LA":
        return 16 if model == "pdformer" or horizon >= 288 else 32
    if dataset == "PEMS08":
        return 16 if model == "pdformer" or horizon >= 288 else 32
    return 8


def default_seq_len(model: str, dataset: str, horizon: int) -> int:
    if model == "lstnn" and horizon >= 288:
        return 96
    return 12


def build_job_specs(args: argparse.Namespace) -> list[JobSpec]:
    root = args.root.resolve()
    package_root = package_root_from_code_root(root)
    run_root = package_root / "results" / "revision_baselines_table3_retrain" / args.run_id
    models = [normalize_model(item) for item in parse_csv_list(args.models, MODELS)]
    datasets = [normalize_dataset(item) for item in parse_csv_list(args.datasets, DATASETS)]
    horizons = parse_int_list(args.horizons, HORIZONS)
    gpus = parse_int_list(args.gpus, tuple(range(8)))
    if not gpus:
        raise ValueError("At least one GPU id is required")

    jobs: list[JobSpec] = []
    gpu_index = 0
    for horizon in horizons:
        for model in models:
            for dataset in datasets:
                if (model, dataset, horizon) in UNSUPPORTED:
                    continue
                batch_size = int(args.batch_size) if int(args.batch_size) > 0 else default_batch_size(model, dataset, horizon)
                seq_len = int(args.seq_len) if int(args.seq_len) > 0 else default_seq_len(model, dataset, int(horizon))
                run_dir = run_root / model / dataset_key(dataset) / f"h{horizon}" / f"seed{args.seed}"
                jobs.append(
                    JobSpec(
                        model=model,
                        dataset=dataset,
                        horizon=int(horizon),
                        seq_len=seq_len,
                        gpu=gpus[gpu_index % len(gpus)],
                        epochs=int(args.epochs),
                        batch_size=batch_size,
                        seed=int(args.seed),
                        run_dir=run_dir,
                    )
                )
                gpu_index += 1
    return jobs


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, object]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as fh:
            existing = list(csv.DictReader(fh))
    key = (str(row.get("model")), str(row.get("dataset")), str(row.get("horizon")))
    filtered = [
        item
        for item in existing
        if (str(item.get("model")), str(item.get("dataset")), str(item.get("horizon"))) != key
    ]
    filtered.append(row)
    write_csv(path, filtered)


def load_raw_series(root: Path, dataset: str) -> np.ndarray:
    data_dir = root / "Datasets" / dataset
    npz_path = data_dir / f"{dataset}.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            raw = data["data"].astype(np.float32)
        if raw.ndim == 2:
            raw = raw[:, :, None]
        return raw[..., :1].astype(np.float32)

    csv_path = data_dir / f"{dataset}.csv"
    if csv_path.exists():
        raw = pd.read_csv(csv_path, header=None).values.astype(np.float32)
        return raw[:, :, None]

    h5_name = "metr-la.h5" if dataset == "METR-LA" else "pems-bay.h5"
    h5_path = data_dir / h5_name
    if h5_path.exists():
        raw = pd.read_hdf(h5_path).values.astype(np.float32)
        return raw[:, :, None]

    raise FileNotFoundError(f"No usable data file for {dataset} under {data_dir}")


def load_node_indices(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    for column in ("matrix_index", "node_index", "index"):
        if column in frame.columns:
            values = frame[column].to_numpy(dtype=np.int64)
            break
    else:
        values = frame.iloc[:, 0].to_numpy(dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Invalid node index file: {path}")
    return values


def append_calendar_features(value: np.ndarray, start_index: int, steps_per_day: int = 288) -> np.ndarray:
    """Append time-of-day and day-of-week channels to a value tensor."""

    if value.ndim != 3 or value.shape[-1] != 1:
        raise ValueError(f"Expected value shape [T,N,1], got {value.shape}")
    t, n, _ = value.shape
    absolute = np.arange(start_index, start_index + t, dtype=np.int64)
    tod = ((absolute % steps_per_day).astype(np.float32) / float(steps_per_day)).reshape(t, 1, 1)
    dow = (((absolute // steps_per_day) % 7).astype(np.float32)).reshape(t, 1, 1)
    tod = np.repeat(tod, n, axis=1)
    dow = np.repeat(dow, n, axis=1)
    return np.concatenate([value.astype(np.float32), tod, dow], axis=-1)


def make_model_arrays(train, val, test, model: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils import MinMaxScaler

    scaler = MinMaxScaler()
    train_value = scaler.fit_transform(train[..., :1]).astype(np.float32)
    val_value = scaler.transform(val[..., :1]).astype(np.float32)
    test_value = scaler.transform(test[..., :1]).astype(np.float32)
    if model == "lstnn":
        train_model = append_calendar_features(train_value, start_index=0)
        val_model = append_calendar_features(val_value, start_index=len(train))
        test_model = append_calendar_features(test_value, start_index=len(train) + len(val))
    else:
        train_model, val_model, test_model = train_value, val_value, test_value
    return train_model, val_model, test_model, scaler


def load_data(
    root: Path,
    dataset: str,
    horizon: int,
    seq_len: int,
    batch_size: int,
    model: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    node_indices_csv: Path | None = None,
    adj_path: Path | None = None,
):
    sys.path.insert(0, str(root / "baseline"))
    from torch.utils.data import DataLoader
    from utils import TrafficDataset, load_adjacency_csv

    raw = load_raw_series(root, dataset)
    node_indices = load_node_indices(node_indices_csv)
    if node_indices is not None and raw.shape[1] != len(node_indices):
        raw = raw[:, node_indices, :]
    total = raw.shape[0]
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError(f"Invalid split ratios: train_ratio={train_ratio}, val_ratio={val_ratio}")
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    train, val, test = raw[:n_train], raw[n_train : n_train + n_val], raw[n_train + n_val :]
    if min(len(train), len(val), len(test)) <= seq_len + horizon:
        raise ValueError(
            f"{dataset} has insufficient split length for horizon={horizon}: "
        f"train={len(train)} val={len(val)} test={len(test)}"
        )
    train_model, val_model, test_model, scaler = make_model_arrays(train, val, test, model)
    print("=" * 60, flush=True)
    print(
        f"数据切分: train={len(train)} ({train_ratio:.3f}), "
        f"val={len(val)} ({val_ratio:.3f}), test={len(test)} ({1.0 - train_ratio - val_ratio:.3f})",
        flush=True,
    )
    print(f"训练集形状: {train_model.shape}", flush=True)
    print(f"训练集 value - 均值: {np.mean(train_model[..., 0]):.4f}, 标准差: {np.std(train_model[..., 0]):.4f}", flush=True)
    print("=" * 60, flush=True)
    loader_kwargs = {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    train_loader = DataLoader(
        TrafficDataset(train_model, seq_len, horizon),
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        TrafficDataset(val_model, seq_len, horizon),
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        TrafficDataset(test_model, seq_len, horizon),
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    print(f"训练批次数: {len(train_loader)}", flush=True)
    print(f"验证批次数: {len(val_loader)}", flush=True)
    print(f"测试批次数: {len(test_loader)}", flush=True)
    print(f"批大小: {batch_size}", flush=True)
    adj_source = adj_path if adj_path is not None else root / "Datasets" / dataset / "adj_matrix.csv"
    adj = load_adjacency_csv(adj_source, normalize="sym", add_self_loops=True)
    if node_indices is not None and adj.shape[0] != len(node_indices):
        import torch

        idx_t = torch.as_tensor(node_indices, dtype=torch.long, device=adj.device)
        adj = adj.index_select(0, idx_t).index_select(1, idx_t)
    return adj, train_loader, val_loader, test_loader, scaler, raw.shape[1], train_model.shape[-1]


def model_config(model: str, epochs: int) -> dict[str, object]:
    if model == "lstnn":
        return {
            "model_kind": "lstnn",
            "hidden_dim": 64,
            "msmlp_patch_len": 12,
            "msmlp_stride": 4,
            "msmlp_d_d": 64,
            "msmlp_d_spa": 16,
            "msmlp_num_layer": 2,
            "lr": 1e-3,
            "patience": min(20, max(8, epochs // 3)),
        }
    if model == "pdformer":
        return {
            "model_kind": "pdformer",
            "hidden_dim": 64,
            "pdf_embed_dim": 64,
            "pdf_skip_dim": 128,
            "pdf_lape_dim": 8,
            "pdf_geo_heads": 4,
            "pdf_sem_heads": 2,
            "pdf_time_heads": 2,
            "pdf_enc_depth": 4,
            "pdf_drop": 0.0,
            "pdf_attn_drop": 0.0,
            "pdf_drop_path": 0.1,
            "lr": 1e-3,
            "patience": min(16, max(6, epochs // 3)),
        }
    raise ValueError(f"Unknown model={model}")


def inverse_with_scaler(tensor, scaler) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    b, t, n, f = arr.shape
    flat = arr.reshape(b * t, n, f)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(b, t, n, f)


def evaluate_table_metrics(model, loader, device, scaler, model_kind: str) -> tuple[dict[str, float], list[dict[str, object]]]:
    import torch
    from train import _forward_bridge

    model.eval()
    totals = {
        "abs": 0.0,
        "sq": 0.0,
        "pct": 0.0,
        "pct_zero_included": 0.0,
        "smape": 0.0,
        "count": 0.0,
        "count_gt0": 0.0,
        "zero_count": 0.0,
    }
    horizon_totals: dict[int, dict[str, float]] = {}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = _forward_bridge(model, x, y, model_kind=model_kind, teacher_forcing_ratio=0.0)
            y_true = y[..., :1]
            horizon = min(y_hat.shape[1], y_true.shape[1])
            y_hat = y_hat[:, :horizon]
            y_true = y_true[:, :horizon]
            pred = inverse_with_scaler(y_hat, scaler)
            true = inverse_with_scaler(y_true, scaler)
            diff = pred - true
            valid = np.isfinite(diff) & np.isfinite(true)
            valid_gt0 = valid & (np.abs(true) > 1e-4)
            abs_err = np.abs(diff)
            sq_err = diff ** 2
            pct_err = abs_err / np.maximum(np.abs(true), 1.0)
            smape_err = abs_err / np.maximum((np.abs(true) + np.abs(pred)) / 2.0, 1.0)

            count = float(valid.sum())
            count_gt0 = float(valid_gt0.sum())
            totals["abs"] += float(abs_err[valid].sum())
            totals["sq"] += float(sq_err[valid].sum())
            totals["pct"] += float(pct_err[valid_gt0].sum())
            totals["pct_zero_included"] += float(pct_err[valid].sum())
            totals["smape"] += float(smape_err[valid].sum())
            totals["count"] += count
            totals["count_gt0"] += count_gt0
            totals["zero_count"] += float((valid & (np.abs(true) <= 1e-4)).sum())

            for h in range(horizon):
                h_valid = valid[:, h]
                h_valid_gt0 = valid_gt0[:, h]
                item = horizon_totals.setdefault(
                    h + 1,
                    {
                        "abs": 0.0,
                        "sq": 0.0,
                        "pct": 0.0,
                        "pct_zero_included": 0.0,
                        "smape": 0.0,
                        "count": 0.0,
                        "count_gt0": 0.0,
                        "zero_count": 0.0,
                    },
                )
                h_count = float(h_valid.sum())
                h_count_gt0 = float(h_valid_gt0.sum())
                item["abs"] += float(abs_err[:, h][h_valid].sum())
                item["sq"] += float(sq_err[:, h][h_valid].sum())
                item["pct"] += float(pct_err[:, h][h_valid_gt0].sum())
                item["pct_zero_included"] += float(pct_err[:, h][h_valid].sum())
                item["smape"] += float(smape_err[:, h][h_valid].sum())
                item["count"] += h_count
                item["count_gt0"] += h_count_gt0
                item["zero_count"] += float((h_valid & (np.abs(true[:, h]) <= 1e-4)).sum())

    if totals["count"] <= 0:
        raise RuntimeError("No valid test observations for metric calculation")

    avg = {
        "mae": totals["abs"] / totals["count"],
        "rmse": float(np.sqrt(totals["sq"] / totals["count"])),
        "mape": (totals["pct"] / totals["count_gt0"]) if totals["count_gt0"] > 0 else float("nan"),
        "mape_zero_included": totals["pct_zero_included"] / totals["count"],
        "smape": totals["smape"] / totals["count"],
        "zero_fraction": totals["zero_count"] / totals["count"],
    }
    per_horizon = []
    for h, item in sorted(horizon_totals.items()):
        if item["count"] <= 0:
            continue
        per_horizon.append(
            {
                "horizon_step": h,
                "mae": item["abs"] / item["count"],
                "rmse": float(np.sqrt(item["sq"] / item["count"])),
                "mape": (item["pct"] / item["count_gt0"]) if item["count_gt0"] > 0 else float("nan"),
                "mape_zero_included": item["pct_zero_included"] / item["count"],
                "smape": item["smape"] / item["count"],
                "count": int(item["count"]),
                "count_gt0": int(item["count_gt0"]),
                "zero_fraction": item["zero_count"] / item["count"],
            }
        )
    return avg, per_horizon


def job_env(gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    cpu_threads = env.get("BASELINE_CPU_THREADS", "4")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = cpu_threads
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    return env


def run_single(args: argparse.Namespace) -> int:
    import random
    import torch

    sys.path.insert(0, str(args.root / "baseline"))
    from train import train_model

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = normalize_model(args.model)
    dataset = normalize_dataset(args.dataset)
    horizon = int(args.horizon)
    seq_len = int(args.seq_len) if int(args.seq_len) > 0 else default_seq_len(model, dataset, horizon)
    if (model, dataset, horizon) in UNSUPPORTED:
        raise ValueError(f"Unsupported cell: {model}/{dataset}/h{horizon}")

    run_dir = Path(args.run_dir) if args.run_dir else (
        package_root_from_code_root(args.root)
        / "results"
        / "revision_baselines_table3_retrain"
        / args.run_id
        / model
        / dataset_key(dataset)
        / f"h{horizon}"
        / f"seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "command.json").open("w", encoding="utf-8") as fh:
        json.dump({k: str(v) for k, v in vars(args).items()}, fh, indent=2, ensure_ascii=False)

    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else default_batch_size(model, dataset, horizon)
    adj, train_loader, val_loader, test_loader, scaler, n_nodes, in_dim = load_data(
        args.root,
        dataset,
        horizon,
        seq_len,
        batch_size,
        model,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        node_indices_csv=args.node_indices_csv,
        adj_path=args.adj_path,
    )
    cfg = model_config(model, int(args.epochs))
    if model == "lstnn" and dataset == "PEMS07" and horizon >= 48:
        cfg["lr"] = min(float(cfg["lr"]), 2e-4)
        os.environ["BASELINE_DISABLE_AMP"] = "1"
    if model == "pdformer":
        cfg["lr"] = min(float(cfg["lr"]), 2e-4)
        os.environ["BASELINE_DISABLE_AMP"] = "1"
    if args.lr_override is not None:
        cfg["lr"] = float(args.lr_override)
    model_kind = str(cfg.pop("model_kind"))
    lr = float(cfg.pop("lr"))
    patience = int(cfg.pop("patience"))
    device = torch.device(args.device)

    start = time.time()
    trained = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        adj=adj,
        n_nodes=n_nodes,
        in_dim=in_dim,
        seq_len=seq_len,
        pred_len=horizon,
        out_dim=1,
        scaler_obj=scaler,
        device=device,
        epochs=int(args.epochs),
        lr=lr,
        patience=patience,
        model_kind=model_kind,
        best_ckpt_path=str(ckpt_dir / "best.pt"),
        last_ckpt_path=str(ckpt_dir / "last.pt"),
        resume_ckpt=str(args.resume_ckpt) if args.resume_ckpt else None,
        resume_states=bool(args.resume_states),
        finetune_lr=float(args.finetune_lr) if args.finetune_lr is not None else None,
        log_txt_path=str(run_dir / "epoch_metrics.tsv"),
        **cfg,
    )
    avg, per_horizon = evaluate_table_metrics(trained, test_loader, device, scaler, model_kind=model_kind)
    elapsed = time.time() - start
    row = {
        "model": "LSTTN" if model == "lstnn" else "PDFormer",
        "model_key": model,
        "dataset": dataset,
        "horizon": horizon,
        "seq_len": seq_len,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": batch_size,
        "lr": f"{lr:.8g}",
        "status": "completed",
        "mae": f"{avg['mae']:.6f}",
        "rmse": f"{avg['rmse']:.6f}",
        "mape": f"{avg['mape']:.6f}",
        "smape": f"{avg['smape']:.6f}",
        "elapsed_sec": f"{elapsed:.1f}",
        "metric_source": "new_retrain_best_val_test_eval",
        "run_dir": str(run_dir),
        "best_ckpt": str(ckpt_dir / "best.pt"),
    }
    write_csv(run_dir / "summary.csv", [row])
    write_csv(run_dir / "per_horizon_metrics.csv", per_horizon)
    with (run_dir / "result.json").open("w", encoding="utf-8") as fh:
        json.dump(row, fh, indent=2, ensure_ascii=False)

    print(
        "Average Horizon, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, sMAPE: {smape:.4f}".format(
            **avg
        ),
        flush=True,
    )
    print(f"Summary: {run_dir / 'summary.csv'}", flush=True)
    return 0


def summary_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        return bool(rows) and all(row.get("status") == "completed" and row.get("mae") for row in rows)
    except Exception:
        return False


def launch_queue(args: argparse.Namespace) -> int:
    jobs = build_job_specs(args)
    run_root = package_root_from_code_root(args.root) / "results" / "revision_baselines_table3_retrain" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    write_csv(run_root / "queue_manifest.csv", [{**asdict(job), "run_dir": str(job.run_dir)} for job in jobs])

    script = Path(__file__).resolve()
    python = Path(args.python).resolve()
    pending = [job for job in jobs if not summary_completed(job.run_dir / "summary.csv")]
    running: dict[subprocess.Popen, JobSpec] = {}
    queue_log = (run_root / "queue.log").open("a", encoding="utf-8", buffering=1)
    queue_log.write(f"[queue] start={time.strftime('%F %T')} pending={len(pending)} total={len(jobs)}\n")

    def start_job(job: JobSpec) -> None:
        job.run_dir.mkdir(parents=True, exist_ok=True)
        log_path = job.run_dir / "run.log"
        cmd = [
            str(python),
            "-u",
            str(script),
            "--mode",
            "single",
            "--root",
            str(args.root),
            "--run_id",
            args.run_id,
            "--model",
            job.model,
            "--dataset",
            job.dataset,
            "--horizon",
            str(job.horizon),
            "--seq_len",
            str(job.seq_len),
            "--epochs",
            str(job.epochs),
            "--batch_size",
            str(job.batch_size),
            "--seed",
            str(job.seed),
            "--device",
            "cuda:0",
            "--run_dir",
            str(job.run_dir),
        ]
        with log_path.open("w", encoding="utf-8") as log_fh:
            log_fh.write(f"[table3] start={time.strftime('%F %T')} gpu={job.gpu} cmd={' '.join(cmd)}\n")
        log_fh = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=job_env(job.gpu), stdout=log_fh, stderr=subprocess.STDOUT, text=True)
        running[proc] = job
        queue_log.write(
            f"[queue] launch={time.strftime('%F %T')} pid={proc.pid} gpu={job.gpu} "
            f"{job.model}/{job.dataset}/h{job.horizon} batch={job.batch_size}\n"
        )

    gpu_pool = parse_int_list(args.gpus, tuple(range(8)))
    max_parallel = min(len(gpu_pool), int(args.max_parallel))
    while pending or running:
        while pending and len(running) < max_parallel:
            busy_gpus = {job.gpu for job in running.values()}
            free_gpus = [gpu for gpu in gpu_pool if gpu not in busy_gpus]
            if not free_gpus:
                break
            start_job(replace(pending.pop(0), gpu=free_gpus[0]))

        finished = []
        for proc, job in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            finished.append(proc)
            status = "completed" if code == 0 and summary_completed(job.run_dir / "summary.csv") else f"failed_returncode_{code}"
            queue_log.write(
                f"[queue] finish={time.strftime('%F %T')} status={status} gpu={job.gpu} "
                f"{job.model}/{job.dataset}/h{job.horizon}\n"
            )
            row = {
                "model": "LSTTN" if job.model == "lstnn" else "PDFormer",
                "model_key": job.model,
                "dataset": job.dataset,
                "horizon": job.horizon,
                "status": status,
                "run_dir": str(job.run_dir),
            }
            summary_path = job.run_dir / "summary.csv"
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8", newline="") as fh:
                    rows = list(csv.DictReader(fh))
                if rows:
                    row.update(rows[0])
            append_csv(run_root / "summary.csv", row)
        for proc in finished:
            running.pop(proc, None)
        if running:
            time.sleep(10)

    queue_log.write(f"[queue] done={time.strftime('%F %T')}\n")
    queue_log.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["single", "queue"], default="single")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--run_id", type=str, default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--models", type=str, default="lstnn,pdformer")
    parser.add_argument("--datasets", type=str, default="METR-LA,PEMS-BAY,PEMS03,PEMS07,PEMS08")
    parser.add_argument("--horizons", type=str, default="12,48,96,288,864,2016")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max_parallel", type=int, default=8)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--node_indices_csv", type=Path)
    parser.add_argument("--adj_path", type=Path)
    parser.add_argument("--resume_ckpt", type=Path)
    parser.add_argument("--resume_states", action="store_true")
    parser.add_argument("--finetune_lr", type=float)
    parser.add_argument("--lr_override", type=float)
    args = parser.parse_args()

    if args.mode == "queue":
        return launch_queue(args)
    if not args.model or not args.dataset:
        parser.error("--model and --dataset are required in single mode")
    return run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
