#!/usr/bin/env python
"""Run auditable Table IV geographic-scale experiments.

The original Table IV uses 1-day forecasting (H=288) on 10/20/40 km
subgraphs.  This runner rebuilds the same node-count cells, stores one
manifest per subgraph, and launches STMAN plus the five revised baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_ORDER = ("PEMS03", "PEMS07", "PEMS08", "PEMS-BAY", "METR-LA")
MODEL_ORDER = ("stman", "graphwavenet", "dmfgcrn", "mtegcrn", "pdformer", "lstnn")
TABLE4_SCALES = {
    "PEMS03": [(10, 126), (20, 268), (40, 328)],
    "PEMS07": [(10, 139), (20, 529), (40, 833)],
    "PEMS08": [(10, 48), (20, 142), (40, 170)],
    "PEMS-BAY": [(10, 86), (20, 247), (40, 325)],
    "METR-LA": [(10, 47), (20, 127), (40, 207)],
}
DATASET_START = {
    "PEMS03": "2018-09-01",
    "PEMS07": "2017-05-01",
    "PEMS08": "2016-07-01",
    "PEMS-BAY": "2017-01-01",
    "METR-LA": "2012-03-01",
}
RECURRENT_DATASET_KEY = {
    "PEMS03": "PEMSD3",
    "PEMS07": "PEMSD7",
    "PEMS08": "PEMSD8",
    "PEMS-BAY": "PEMSBAY",
    "METR-LA": "METRLA",
}


@dataclass(frozen=True)
class SubgraphSpec:
    dataset: str
    radius_km: int
    nodes: int
    key: str
    node_csv: Path
    series_npz: Path
    adj_csv: Path
    gwn_adj_pkl: Path
    recurrent_name: str


@dataclass(frozen=True)
class JobSpec:
    model: str
    dataset: str
    radius_km: int
    nodes: int
    gpu: int
    epochs: int
    batch_size: int
    run_dir: Path
    command: list[str]
    cwd: Path
    subgraph_key: str
    recurrent_name: str = ""


def package_root_from_code_root(root: Path) -> Path:
    root = root.resolve()
    if root.name != "multifract":
        raise ValueError(f"Expected code root ending in multifract, got {root}")
    return root.parents[1]


def dataset_key(dataset: str) -> str:
    return dataset.lower().replace("-", "").replace("_", "")


def parse_csv_list(text: str, default: tuple[str, ...]) -> list[str]:
    if not text:
        return list(default)
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_int_csv_list(text: str, default: tuple[int, ...]) -> list[int]:
    if not text:
        return list(default)
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def upsert_csv(path: Path, row: dict[str, object], key_fields: tuple[str, ...]) -> None:
    rows: list[dict[str, object]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    key = tuple(str(row.get(field, "")) for field in key_fields)
    rows = [
        item for item in rows
        if tuple(str(item.get(field, "")) for field in key_fields) != key
    ]
    rows.append(row)
    write_csv(path, rows)


def load_raw_series(code_root: Path, dataset: str) -> np.ndarray:
    data_dir = code_root / "Datasets" / dataset
    npz_path = data_dir / f"{dataset}.npz"
    if npz_path.exists():
        raw = np.load(npz_path)["data"].astype(np.float32)
    elif dataset == "METR-LA" and (data_dir / "metr-la.h5").exists():
        try:
            raw = pd.read_hdf(data_dir / "metr-la.h5").values.astype(np.float32)
        except ImportError:
            raw = pd.read_csv(data_dir / "METR-LA.csv", header=None).values.astype(np.float32)
    elif dataset == "PEMS-BAY" and (data_dir / "pems-bay.h5").exists():
        try:
            raw = pd.read_hdf(data_dir / "pems-bay.h5").values.astype(np.float32)
        except ImportError:
            raw = pd.read_csv(data_dir / "PEMS-BAY.csv", header=None).values.astype(np.float32)
    else:
        raw = pd.read_csv(data_dir / f"{dataset}.csv", header=None).values.astype(np.float32)
    if raw.ndim == 2:
        raw = raw[:, :, None]
    return raw[..., :1].astype(np.float32)


def load_adjacency(code_root: Path, dataset: str) -> np.ndarray:
    return pd.read_csv(code_root / "Datasets" / dataset / "adj_matrix.csv", header=None).values.astype(np.float32)


def load_locations(code_root: Path, dataset: str) -> pd.DataFrame | None:
    if dataset == "METR-LA":
        path = code_root / "Datasets" / dataset / "graph_sensor_locations_metrLa.csv"
        if path.exists():
            return pd.read_csv(path)
    if dataset == "PEMS-BAY":
        path = code_root / "Datasets" / dataset / "graph_sensor_locations_bay.csv"
        if path.exists():
            return pd.read_csv(path, header=None, names=["sensor_id", "latitude", "longitude"])
    return None


def equirectangular_xy_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat0 = np.deg2rad(float(np.mean(lat)))
    x = np.deg2rad(lon) * np.cos(lat0) * 6371.0088
    y = np.deg2rad(lat) * 6371.0088
    return np.column_stack([x, y])


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(set(map(tuple, points.tolist())))
    if len(pts) <= 1:
        return np.asarray(pts, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def geometry_summary(locations: pd.DataFrame | None, indices: np.ndarray) -> dict[str, object]:
    if locations is None or not {"latitude", "longitude"}.issubset(locations.columns):
        return {
            "extent_source": "prefix_node_order; coordinate_file_unavailable",
            "convex_hull_area_km2": "",
            "geographic_diameter_km": "",
        }
    loc = locations.iloc[indices]
    xy = equirectangular_xy_km(loc["latitude"].to_numpy(float), loc["longitude"].to_numpy(float))
    hull = convex_hull(xy)
    if len(hull) >= 3:
        area = 0.5 * abs(np.dot(hull[:, 0], np.roll(hull[:, 1], -1)) - np.dot(hull[:, 1], np.roll(hull[:, 0], -1)))
    else:
        area = 0.0
    if len(xy) >= 2:
        diff = xy[:, None, :] - xy[None, :, :]
        diameter = float(np.sqrt(np.sum(diff * diff, axis=-1)).max())
    else:
        diameter = 0.0
    return {
        "extent_source": "prefix_node_order_with_coordinates",
        "convex_hull_area_km2": f"{area:.6f}",
        "geographic_diameter_km": f"{diameter:.6f}",
    }


def write_gwn_adj_pickle(adj: np.ndarray, output_path: Path) -> None:
    adj_bin = (adj > 0).astype(np.float32)
    np.fill_diagonal(adj_bin, 1.0)
    sensor_ids = [str(i) for i in range(adj_bin.shape[0])]
    sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        pickle.dump((sensor_ids, sensor_id_to_ind, adj_bin), fh, protocol=pickle.HIGHEST_PROTOCOL)


def write_recurrent_config(model_dir: Path, model_name: str, dataset_name: str, nodes: int, horizon: int, batch_size: int) -> Path:
    path = model_dir / "config_file" / f"{dataset_name}_{model_name}.conf"
    embed_dim = 8
    text = f"""[data]
num_nodes = {nodes}
lag = 12
horizon = {horizon}
val_ratio = 0.2
test_ratio = 0.2
tod = False
normalizer = std
column_wise = False
default_graph = True
add_time_in_day=True
add_day_in_week=True
steps_per_day=288

[model]
input_dim = 1
output_dim = 1
embed_dim = {embed_dim}
rnn_units = 64
num_layers = 1
cheb_order = 2
use_day = True
use_week = True

[train]
loss_func = mae
seed = 42
batch_size = {batch_size}
epochs = 20
lr_init = 0.003
weight_decay = 0
lr_decay = False
lr_decay_rate = 0.1
lr_decay_step = 5,20,40,70
early_stop = False
early_stop_patience = 10
grad_norm = False
max_grad_norm = 5
real_value = True

[test]
mae_thresh = None
mape_thresh = 0.

[log]
log_step = 2000
plot = False
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def prepare_subgraphs(args: argparse.Namespace) -> list[SubgraphSpec]:
    code_root = args.root.resolve()
    package_root = package_root_from_code_root(code_root)
    run_root = package_root / "results" / "table4_geo_scale" / args.run_id
    prepared_root = run_root / "prepared"
    radii_km = set(parse_int_csv_list(args.radii_km, (10, 20, 40)))
    rows: list[dict[str, object]] = []
    specs: list[SubgraphSpec] = []
    for dataset in parse_csv_list(args.datasets, DATASET_ORDER):
        raw = load_raw_series(code_root, dataset)
        full_adj = load_adjacency(code_root, dataset)
        locations = load_locations(code_root, dataset)
        for radius_km, nodes in TABLE4_SCALES[dataset]:
            if radius_km not in radii_km:
                continue
            if nodes > raw.shape[1] or nodes > full_adj.shape[0]:
                raise ValueError(f"{dataset} radius={radius_km} nodes={nodes} exceeds raw/adj shape")
            indices = np.arange(nodes, dtype=np.int64)
            key = f"{dataset_key(dataset)}_r{radius_km}km_n{nodes}"
            cell_dir = prepared_root / key
            node_csv = cell_dir / "node_indices.csv"
            series_npz = cell_dir / "series.npz"
            adj_csv = cell_dir / "adj_matrix.csv"
            gwn_adj_pkl = cell_dir / "adj_mx.pkl"
            cell_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"matrix_index": indices}).to_csv(node_csv, index=False)
            sub_raw = raw[:, indices, :]
            sub_adj = full_adj[np.ix_(indices, indices)]
            np.savez_compressed(series_npz, data=sub_raw)
            np.savetxt(adj_csv, sub_adj, delimiter=",", fmt="%.6f")
            write_gwn_adj_pickle(sub_adj, gwn_adj_pkl)
            recurrent_name = f"T4_{dataset_key(dataset).upper()}_R{radius_km}_N{nodes}"
            recurrent_dir = prepared_root / "recurrent_data" / recurrent_name
            recurrent_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(recurrent_dir / f"{recurrent_name}.npz", data=sub_raw)
            geom = geometry_summary(locations, indices)
            row = {
                "dataset": dataset,
                "radius_km": radius_km,
                "nodes": nodes,
                "subgraph_key": key,
                "node_selection": "first_K_nodes_reproducing_original_table4_node_counts",
                "node_csv": str(node_csv),
                "series_npz": str(series_npz),
                "adj_csv": str(adj_csv),
                "gwn_adj_pkl": str(gwn_adj_pkl),
                "recurrent_name": recurrent_name,
                "raw_shape": list(raw.shape),
                "sub_raw_shape": list(sub_raw.shape),
                **geom,
            }
            rows.append(row)
            (cell_dir / "manifest.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
            specs.append(SubgraphSpec(dataset, radius_km, nodes, key, node_csv, series_npz, adj_csv, gwn_adj_pkl, recurrent_name))
    write_csv(run_root / "subgraph_manifest.csv", rows)
    return specs


def default_batch_size(model: str, dataset: str, nodes: int) -> int:
    if model in {"pdformer", "lstnn"}:
        return max(1, min(16, 32768 // max(nodes, 1)))
    if model == "graphwavenet":
        return max(1, min(32, 16384 // max(nodes, 1)))
    if model in {"dmfgcrn", "mtegcrn"}:
        return max(1, min(32, 8192 // max(nodes, 1)))
    if model == "stman":
        return max(1, min(8, 4096 // max(nodes, 1)))
    return 4


def build_jobs(args: argparse.Namespace, specs: list[SubgraphSpec]) -> list[JobSpec]:
    code_root = args.root.resolve()
    package_root = package_root_from_code_root(code_root)
    run_root = package_root / "results" / "table4_geo_scale" / args.run_id
    python = str(args.python.resolve())
    models = parse_csv_list(args.models, MODEL_ORDER)
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("No GPUs specified")
    jobs: list[JobSpec] = []
    job_index = 0
    for spec in specs:
        for model in models:
            gpu = gpus[job_index % len(gpus)]
            batch_size = default_batch_size(model, spec.dataset, spec.nodes)
            epochs = int(args.epochs)
            run_dir = run_root / "runs" / model / spec.key
            if model in {"lstnn", "pdformer"}:
                command = [
                    python, "-u", str(code_root / "baseline" / "train_table3_baseline.py"),
                    "--mode", "single",
                    "--root", str(code_root),
                    "--run_id", args.run_id,
                    "--model", model,
                    "--dataset", spec.dataset,
                    "--horizon", str(args.horizon),
                    "--epochs", str(epochs),
                    "--batch_size", str(batch_size),
                    "--seed", str(args.seed),
                    "--device", "cuda:0",
                    "--run_dir", str(run_dir),
                    "--node_indices_csv", str(spec.node_csv),
                    "--adj_path", str(spec.adj_csv),
                ]
                cwd = code_root
            elif model == "graphwavenet":
                command = [
                    python, "-u", "train.py",
                    "--device", "cuda:0",
                    "--data", str(run_dir / "unused_prepared_npz"),
                    "--adjdata", str(spec.gwn_adj_pkl),
                    "--adjtype", "doubletransition",
                    "--gcn_bool", "--addaptadj", "--randomadj",
                    "--seq_length", str(args.horizon),
                    "--nhid", "32",
                    "--in_dim", "2",
                    "--num_nodes", str(spec.nodes),
                    "--batch_size", str(batch_size),
                    "--learning_rate", "0.001",
                    "--dropout", "0.3",
                    "--weight_decay", "0.0001",
                    "--epochs", str(epochs),
                    "--print_every", "100",
                    "--save", str(run_dir / "checkpoints" / "gwn"),
                    "--expid", str(args.seed),
                    "--stream_source", str(spec.series_npz),
                    "--stream_seq_length_x", "12",
                    "--stream_train_ratio", "0.6",
                    "--stream_val_ratio", "0.2",
                    "--stream_test_ratio", "0.2",
                ]
                cwd = code_root / "baseline" / "Graph-WaveNet"
            elif model in {"dmfgcrn", "mtegcrn"}:
                model_dir_name = "DMFGCRN" if model == "dmfgcrn" else "MTEGCRN"
                model_dir = code_root / "baseline" / model_dir_name
                write_recurrent_config(model_dir, model_dir_name, spec.recurrent_name, spec.nodes, args.horizon, batch_size)
                command = [
                    python, "-u", "run_single.py",
                    "--dataset", spec.recurrent_name,
                    "--model", model_dir_name,
                    "--device", "cuda:0",
                    "--seed", str(args.seed),
                    "--epochs", str(epochs),
                    "--horizon", str(args.horizon),
                    "--val_ratio", "0.2",
                    "--test_ratio", "0.2",
                    "--loss_func", "mae",
                    "--batch_size", str(batch_size),
                    "--early_stop", "False",
                ]
                cwd = model_dir
            elif model == "stman":
                progress = run_dir / "progress.log"
                command = [
                    python, "-u", str(code_root / "train_metrla_optimized.py"),
                    "--dataset", spec.dataset,
                    "--seq_len", "12",
                    "--pred_len", str(args.horizon),
                    "--d_model", "64",
                    "--n_heads", "4",
                    "--enc_layers", "3",
                    "--dec_layers", "2",
                    "--batch_size", str(batch_size),
                    "--epochs", str(epochs),
                    "--lr", "0.0005",
                    "--patience", "8",
                    "--split_rate", "0.6",
                    "--val_ratio", "0.2",
                    "--scaler", "standard",
                    "--tag", f"table4_{spec.key}",
                    "--progress_log", str(progress),
                    "--log_interval", "1",
                    "--use_direct_pred",
                    "--direct_head_mode", "attn",
                    "--direct_step_refine", "1",
                    "--spatial_mode", "hybrid",
                    "--k_hop", "8",
                    "--add_dow_feature",
                    "--scale_target_only",
                    "--gated_residual",
                    "--node_indices_csv", str(spec.node_csv),
                    "--adj_path", str(spec.adj_csv),
                    "--checkpoint_dir", str(run_dir / "checkpoints"),
                ]
                cwd = code_root
            else:
                raise ValueError(f"Unknown model: {model}")
            jobs.append(JobSpec(model, spec.dataset, spec.radius_km, spec.nodes, gpu, epochs, batch_size, run_dir, command, cwd, spec.key, spec.recurrent_name))
            job_index += 1
    return jobs


def job_env(job: JobSpec, run_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    env["HIP_VISIBLE_DEVICES"] = str(job.gpu)
    env["TABLE4_CUSTOM_DATA_ROOT"] = str(run_root / "prepared" / "recurrent_data")
    env.setdefault("BASELINE_SAVE_NUMPY_ARTIFACTS", "0")
    threads = env.get("BASELINE_CPU_THREADS", "4")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = threads
    return env


def parse_metrics(job: JobSpec) -> dict[str, str]:
    log_path = job.run_dir / "run.log"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if job.model in {"lstnn", "pdformer"}:
        result_path = job.run_dir / "result.json"
        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            return {
                "test_mae_best_val": str(data.get("mae", "")),
                "test_rmse_best_val": str(data.get("rmse", "")),
                "test_mape_best_val": str(data.get("mape", "")),
                "metric_source": "new_retrain_best_val_test_eval",
            }
    if job.model == "graphwavenet":
        match = re.search(r"On average over \d+ horizons, Test MAE: ([0-9.]+), Test MAPE: ([0-9.]+), Test RMSE: ([0-9.]+)", text)
        if match:
            return {
                "test_mae_best_val": match.group(1),
                "test_mape_best_val": match.group(2),
                "test_rmse_best_val": match.group(3),
                "metric_source": "graphwavenet_streaming_test_eval",
            }
    if job.model in {"dmfgcrn", "mtegcrn"}:
        if "=== Best validation model results ===" in text:
            block = text.split("=== Best validation model results ===", 1)[1]
            if "=== Best test model results ===" in block:
                block = block.split("=== Best test model results ===", 1)[0]
            match = re.search(r"Average Horizon, MAE: ([0-9.]+), RMSE: ([0-9.]+), MAPE: ([0-9.]+)", block)
            if match:
                return {
                    "test_mae_best_val": match.group(1),
                    "test_rmse_best_val": match.group(2),
                    "test_mape_best_val": match.group(3),
                    "metric_source": "recurrent_best_val_test_eval",
                }
    if job.model == "stman":
        matches = re.findall(r"best_MAE=([0-9.]+)", text)
        progress_path = job.run_dir / "progress.log"
        if progress_path.exists():
            matches.extend(re.findall(r"best_MAE=([0-9.]+)", progress_path.read_text(encoding="utf-8", errors="replace")))
        if matches:
            return {
                "val_mae_best": f"{min(float(x) for x in matches):.6f}",
                "metric_source": "stman_validation_progress_pending_test_eval",
            }
    return {}


def launch(args: argparse.Namespace) -> int:
    specs = prepare_subgraphs(args)
    jobs = build_jobs(args, specs)
    run_root = package_root_from_code_root(args.root.resolve()) / "results" / "table4_geo_scale" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    write_csv(run_root / "queue_manifest.csv", [{**asdict(job), "run_dir": str(job.run_dir), "cwd": str(job.cwd), "command": " ".join(job.command)} for job in jobs])
    if args.dry_run:
        for job in jobs:
            print(job.gpu, job.model, job.dataset, job.radius_km, job.nodes, " ".join(job.command))
        return 0

    pending = [job for job in jobs if not (job.run_dir / "metrics.csv").exists()]
    running: dict[subprocess.Popen, tuple[JobSpec, object, float]] = {}
    queue_log = (run_root / "queue.log").open("a", encoding="utf-8", buffering=1)
    queue_log.write(f"[table4] start={time.strftime('%F %T')} pending={len(pending)} total={len(jobs)}\n")
    gpu_pool = [int(item) for item in args.gpus.split(",") if item.strip()]
    max_parallel = min(int(args.max_parallel), len(gpu_pool))

    while pending or running:
        while pending and len(running) < max_parallel:
            busy_gpus = {active_job.gpu for active_job, _, _ in running.values()}
            free_gpus = [gpu for gpu in gpu_pool if gpu not in busy_gpus]
            if not free_gpus:
                break
            job = replace(pending.pop(0), gpu=free_gpus[0])
            job.run_dir.mkdir(parents=True, exist_ok=True)
            (job.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (job.run_dir / "command.json").write_text(
                json.dumps({**asdict(job), "cwd": str(job.cwd), "run_dir": str(job.run_dir)}, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            log_fh = (job.run_dir / "run.log").open("w", encoding="utf-8", buffering=1)
            log_fh.write(
                f"[table4] start={time.strftime('%F %T')} model={job.model} dataset={job.dataset} "
                f"radius={job.radius_km} nodes={job.nodes} gpu={job.gpu}\n"
            )
            log_fh.write("[table4] command=" + " ".join(job.command) + "\n")
            proc = subprocess.Popen(job.command, cwd=job.cwd, env=job_env(job, run_root), stdout=log_fh, stderr=subprocess.STDOUT, text=True)
            running[proc] = (job, log_fh, time.time())
            queue_log.write(f"[table4] launch={time.strftime('%F %T')} pid={proc.pid} gpu={job.gpu} {job.model}/{job.subgraph_key}\n")

        for proc, (job, log_fh, start_time) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log_fh.write(f"\n[table4] end={time.strftime('%F %T')} returncode={code}\n")
            log_fh.close()
            status = "completed" if code == 0 else f"failed_returncode_{code}"
            parsed = parse_metrics(job)
            row = {
                "model": job.model,
                "dataset": job.dataset,
                "radius_km": job.radius_km,
                "nodes": job.nodes,
                "horizon": args.horizon,
                "epochs": job.epochs,
                "batch_size": job.batch_size,
                "gpu": job.gpu,
                "status": status,
                "returncode": code,
                "elapsed_sec": f"{time.time() - start_time:.1f}",
                "run_dir": str(job.run_dir),
                "log_path": str(job.run_dir / "run.log"),
                **parsed,
            }
            write_csv(job.run_dir / "metrics.csv", [row])
            upsert_csv(run_root / "summary.csv", row, key_fields=("model", "dataset", "radius_km", "nodes"))
            queue_log.write(f"[table4] finish={time.strftime('%F %T')} status={status} gpu={job.gpu} {job.model}/{job.subgraph_key}\n")
            running.pop(proc)
        if running:
            time.sleep(10)
    queue_log.write(f"[table4] done={time.strftime('%F %T')}\n")
    queue_log.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--run_id", type=str, default=time.strftime("%Y%m%d_table4_geo_scale"))
    parser.add_argument("--datasets", type=str, default=",".join(DATASET_ORDER))
    parser.add_argument("--models", type=str, default=",".join(MODEL_ORDER))
    parser.add_argument("--radii_km", type=str, default="10,20,40")
    parser.add_argument("--gpus", type=str, default="4,5,6")
    parser.add_argument("--max_parallel", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=288)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
