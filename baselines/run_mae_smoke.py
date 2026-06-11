#!/usr/bin/env python
"""Run a small MAE-only baseline smoke queue for the revision package.

This is intentionally small: it validates the result path and training chain
before launching the full 90-run grid.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SmokeJob:
    model: str
    dataset: str
    internal_dataset: str
    horizon: int
    seed: int
    gpu: int
    epochs: int
    cwd: Path
    run_dir: Path
    command: list[str]


def package_root_from_code_root(root: Path) -> Path:
    root = root.resolve()
    if root.name != "multifract":
        raise ValueError(f"Expected code root ending in multifract, got {root}")
    return root.parents[1]


def _run_dir(root: Path, model: str, dataset: str, horizon: int, seed: int, run_suffix: str = "") -> Path:
    package_root = package_root_from_code_root(root)
    dataset_key = dataset.lower().replace("-", "")
    suffix = f"_{run_suffix.strip('_')}" if run_suffix else ""
    return package_root / "results" / "revision_smoke" / f"{model}_{dataset_key}_h{horizon}_seed{seed}{suffix}"


def build_smoke_jobs(
    root: Path,
    gpu: int,
    epochs: int,
    seed: int,
    models: list[str] | None = None,
    recurrent_batch_size: int = 64,
    run_suffix: str = "",
) -> list[SmokeJob]:
    if gpu == 2:
        raise ValueError("GPU2 is intentionally left unused for the CAD workload.")

    selected = set(models or ["graphwavenet", "dmfgcrn", "mtegcrn"])
    python = os.environ.get("PYTHON", sys.executable)
    baseline_root = root / "baseline"
    jobs: list[SmokeJob] = []

    if "graphwavenet" in selected:
        gwn_dir = baseline_root / "Graph-WaveNet"
        gwn_run_dir = _run_dir(root, "graphwavenet", "METR-LA", 12, seed, run_suffix=run_suffix)
        gwn_data_dir = package_root_from_code_root(root) / "results" / "revision_smoke" / "mini_data" / "graphwavenet_metrla_h12"
        jobs.append(
            SmokeJob(
                model="graphwavenet",
                dataset="METR-LA",
                internal_dataset="METR-LA",
                horizon=12,
                seed=seed,
                gpu=gpu,
                epochs=epochs,
                cwd=gwn_dir,
                run_dir=gwn_run_dir,
                command=[
                    python,
                    "-u",
                    "train.py",
                    "--device",
                    "cuda:0",
                    "--data",
                    str(gwn_data_dir),
                    "--adjdata",
                    "data/sensor_graph/adj_mx.pkl",
                    "--adjtype",
                    "doubletransition",
                    "--gcn_bool",
                    "--addaptadj",
                    "--randomadj",
                    "--seq_length",
                    "12",
                    "--nhid",
                    "16",
                    "--in_dim",
                    "2",
                    "--num_nodes",
                    "207",
                    "--batch_size",
                    str(recurrent_batch_size),
                    "--learning_rate",
                    "0.001",
                    "--dropout",
                    "0.3",
                    "--weight_decay",
                    "0.0001",
                    "--epochs",
                    str(epochs),
                    "--print_every",
                    "100",
                    "--save",
                    str(gwn_run_dir / "checkpoints" / "gwn_metrla"),
                    "--expid",
                    str(seed),
                ],
            )
        )

    for model, model_dir_name in [("dmfgcrn", "DMFGCRN"), ("mtegcrn", "MTEGCRN")]:
        if model not in selected:
            continue
        model_dir = baseline_root / model_dir_name
        run_dir = _run_dir(root, model, "PEMS08", 12, seed, run_suffix=run_suffix)
        jobs.append(
            SmokeJob(
                model=model,
                dataset="PEMS08",
                internal_dataset="PEMSD8",
                horizon=12,
                seed=seed,
                gpu=gpu,
                epochs=epochs,
                cwd=model_dir,
                run_dir=run_dir,
                command=[
                    python,
                    "-u",
                    "run_single.py",
                    "--dataset",
                    "PEMSD8",
                    "--model",
                    model_dir_name,
                    "--device",
                    "cuda:0",
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(epochs),
                    "--horizon",
                    "12",
                    "--loss_func",
                    "mae",
                    "--batch_size",
                    str(recurrent_batch_size),
                    "--early_stop",
                    "False",
                ],
            )
        )

    return jobs


def ensure_graphwavenet_adjacency(root: Path) -> None:
    sensor_dir = root / "baseline" / "Graph-WaveNet" / "data" / "sensor_graph"
    sensor_dir.mkdir(parents=True, exist_ok=True)
    links = {
        "adj_mx.pkl": root / "Datasets" / "METR-LA" / "adj_mat.pkl",
        "adj_mx_bay.pkl": root / "Datasets" / "PEMS-BAY" / "adj_mx_bay.pkl",
    }
    for name, source in links.items():
        target = sensor_dir / name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(os.path.relpath(source, sensor_dir))


def prepare_graphwavenet_mini_data(root: Path, max_train: int = 512, max_val: int = 128, max_test: int = 128) -> Path:
    source_dir = root / "baseline" / "Graph-WaveNet" / "data" / "METR-LA"
    target_dir = package_root_from_code_root(root) / "results" / "revision_smoke" / "mini_data" / "graphwavenet_metrla_h12"
    target_dir.mkdir(parents=True, exist_ok=True)
    limits = {"train": max_train, "val": max_val, "test": max_test}
    import numpy as np

    for split, limit in limits.items():
        source = source_dir / f"{split}.npz"
        target = target_dir / f"{split}.npz"
        if target.exists():
            continue
        data = np.load(source)
        arrays = {}
        for key in data.files:
            value = data[key]
            if key in {"x", "y"}:
                arrays[key] = value[:limit]
            else:
                arrays[key] = value
        np.savez_compressed(target, **arrays)
    return target_dir


def parse_metrics(log_path: Path) -> dict[str, str]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    val_losses = [float(x) for x in re.findall(r"Val Epoch \d+: average Loss: ([0-9.]+)", text)]
    gwn_avg = re.findall(r"On average over \d+ horizons, Test MAE: ([0-9.]+)", text)
    avg_horizon = re.findall(r"Average Horizon, MAE: ([0-9.]+)", text)
    recurrent_test = re.findall(r"test Epoch \d+: average Loss: ([0-9.]+)", text, flags=re.IGNORECASE)

    test_maes = [float(x) for x in gwn_avg + avg_horizon + recurrent_test]
    result = {
        "val_mae_min": f"{min(val_losses):.6f}" if val_losses else "",
        "test_mae_best_val": f"{test_maes[0]:.6f}" if test_maes else "",
        "test_mae_min_logged": f"{min(test_maes):.6f}" if test_maes else "",
    }
    return result


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upsert_summary_row(path: Path, row: dict[str, str], key: str = "run_dir") -> None:
    """Update one summary row without dropping rows written by other runners."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)

        replaced = False
        for index, existing in enumerate(rows):
            if existing.get(key) == row.get(key):
                rows[index] = {**existing, **row}
                replaced = True
                break
        if not replaced:
            rows.append(row)

        for existing in rows:
            for name in existing:
                if name not in fieldnames:
                    fieldnames.append(name)

        fh.seek(0)
        fh.truncate()
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        fcntl.flock(fh, fcntl.LOCK_UN)


def copy_prediction_artifacts(job: SmokeJob) -> None:
    for suffix in ["pred", "true"]:
        source = job.cwd / f"{job.internal_dataset}_{suffix}.npy"
        if source.exists():
            shutil.copy2(source, job.run_dir / f"{suffix}.npy")


def build_job_env(job: SmokeJob, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    return env


def run_job(job: SmokeJob, timeout_minutes: int) -> dict[str, str]:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    (job.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with (job.run_dir / "command.json").open("w", encoding="utf-8") as fh:
        json.dump({**asdict(job), "cwd": str(job.cwd), "run_dir": str(job.run_dir)}, fh, indent=2, default=str)

    log_path = job.run_dir / "run.log"
    env = build_job_env(job)

    start = time.time()
    status = "completed"
    returncode = 0
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[smoke] start={time.strftime('%F %T')} model={job.model} dataset={job.dataset} horizon={job.horizon} gpu={job.gpu}\n")
        log.write("[smoke] command=" + " ".join(job.command) + "\n")
        log.flush()
        try:
            proc = subprocess.run(
                job.command,
                cwd=job.cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_minutes * 60,
                check=False,
            )
            returncode = proc.returncode
            if proc.returncode != 0:
                status = "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
            returncode = 124
            log.write(f"\n[smoke] timeout after {timeout_minutes} minutes\n")
        log.write(f"\n[smoke] end={time.strftime('%F %T')} status={status} returncode={returncode}\n")

    copy_prediction_artifacts(job)
    parsed = parse_metrics(log_path)
    row = {
        "model": job.model,
        "dataset": job.dataset,
        "internal_dataset": job.internal_dataset,
        "horizon": str(job.horizon),
        "seed": str(job.seed),
        "gpu": str(job.gpu),
        "epochs": str(job.epochs),
        "status": status,
        "returncode": str(returncode),
        "elapsed_sec": f"{time.time() - start:.1f}",
        "run_dir": str(job.run_dir),
        "log_path": str(log_path),
        **parsed,
    }
    write_csv(job.run_dir / "metrics.csv", [row])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", type=str, default="graphwavenet,dmfgcrn,mtegcrn")
    parser.add_argument("--recurrent_batch_size", type=int, default=64)
    parser.add_argument("--run_suffix", type=str, default="")
    parser.add_argument("--timeout_minutes", type=int, default=45)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    models = [model.strip().lower() for model in args.models.split(",") if model.strip()]
    jobs = build_smoke_jobs(
        root=root,
        gpu=args.gpu,
        epochs=args.epochs,
        seed=args.seed,
        models=models,
        recurrent_batch_size=args.recurrent_batch_size,
        run_suffix=args.run_suffix,
    )
    if any(job.model == "graphwavenet" for job in jobs):
        ensure_graphwavenet_adjacency(root)
        if not args.dry_run:
            prepare_graphwavenet_mini_data(root)

    package_root = package_root_from_code_root(root)
    summary_path = package_root / "results" / "revision_smoke" / "summary.csv"
    if args.dry_run:
        for job in jobs:
            print(job.model, job.dataset, " ".join(job.command))
        return 0

    rows = []
    for job in jobs:
        row = run_job(job, timeout_minutes=args.timeout_minutes)
        rows.append(row)
        upsert_summary_row(summary_path, row)
    return 0 if all(row["status"] == "completed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
