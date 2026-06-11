#!/usr/bin/env python
"""Launch an auditable baseline short-run queue.

The default is still h12 for the revision main table, but the runner also
accepts longer horizons so the same audit format can backfill the full grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Job:
    name: str
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


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
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
    env.setdefault("BASELINE_SAVE_NUMPY_ARTIFACTS", "0")
    return env


def extract_average(block: str) -> tuple[str, str, str]:
    match = re.search(
        r"Average Horizon, MAE: ([0-9.]+), RMSE: ([0-9.]+), MAPE: ([0-9.]+)",
        block,
    )
    if not match:
        return "", "", ""
    return match.group(1), match.group(2), match.group(3)


def parse_log(job: Job, log_path: Path) -> dict[str, str]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    row: dict[str, str] = {}

    exp_match = re.search(r"Log directory: (.+)", text)
    if exp_match:
        row["experiment_dir"] = exp_match.group(1).strip()

    if job.model == "graphwavenet":
        val_match = re.search(r"The valid loss on best model is ([0-9.]+)", text)
        avg_match = re.search(
            r"On average over \d+ horizons, Test MAE: ([0-9.]+), Test MAPE: ([0-9.]+), Test RMSE: ([0-9.]+)",
            text,
        )
        if val_match:
            row["val_mae_best"] = val_match.group(1)
        if avg_match:
            row["test_mae_best_val"] = avg_match.group(1)
            row["test_mape_best_val"] = avg_match.group(2)
            row["test_rmse_best_val"] = avg_match.group(3)
        return row

    val_losses = [float(x) for x in re.findall(r"Val Epoch \d+: average Loss: ([0-9.]+)", text)]
    test_losses = [float(x) for x in re.findall(r"test Epoch \d+: average Loss: ([0-9.]+)", text, re.I)]
    if val_losses:
        row["val_mae_min"] = f"{min(val_losses):.6f}"
    if test_losses:
        row["test_mae_min_logged"] = f"{min(test_losses):.6f}"

    if "=== Best validation model results ===" in text:
        block = text.split("=== Best validation model results ===", 1)[1]
        if "=== Best test model results ===" in block:
            block = block.split("=== Best test model results ===", 1)[0]
        mae, rmse, mape = extract_average(block)
        row["test_mae_best_val"] = mae
        row["test_rmse_best_val"] = rmse
        row["test_mape_best_val"] = mape
    return row


def copy_artifacts(job: Job, *, copy_numpy_artifacts: bool) -> None:
    if not copy_numpy_artifacts:
        return
    for suffix in ["pred", "true"]:
        source = job.cwd / f"{job.internal_dataset}_{suffix}.npy"
        if source.exists():
            shutil.copy2(source, job.run_dir / f"{suffix}.npy")


def build_jobs(args: argparse.Namespace) -> list[Job]:
    root = args.root.resolve()
    package_root = package_root_from_code_root(root)
    baseline_root = root / "baseline"
    run_root = package_root / "results" / "revision_baselines_12step_short" / args.run_id
    python = str(args.python)
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    seed = int(args.seed)
    horizon = int(args.horizon)

    dataset_map = {
        "METR-LA": "METRLA",
        "PEMS-BAY": "PEMSBAY",
        "PEMS03": "PEMSD3",
        "PEMS07": "PEMSD7",
        "PEMS08": "PEMSD8",
    }
    gwn_dataset_map = {"METR-LA": "METR-LA", "PEMS-BAY": "PEMS-BAY", "PEMS03": "PEMS03", "PEMS07": "PEMS07", "PEMS08": "PEMS08"}
    selected_models = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    selected_datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    specs: list[tuple[str, str, str, int]] = []
    for model in selected_models:
        for dataset in selected_datasets:
            if dataset not in dataset_map:
                raise ValueError(f"Unknown dataset {dataset}; choices={sorted(dataset_map)}")
            if model == "graphwavenet":
                specs.append((model, dataset, gwn_dataset_map[dataset], args.gwn_epochs))
            elif model == "dmfgcrn":
                specs.append((model, dataset, dataset_map[dataset], args.dmfg_epochs))
            elif model == "mtegcrn":
                specs.append((model, dataset, dataset_map[dataset], args.mteg_epochs))
            else:
                raise ValueError(f"Unknown model {model}; choices=graphwavenet,dmfgcrn,mtegcrn")

    jobs: list[Job] = []
    for index, (model, dataset, internal_dataset, epochs) in enumerate(specs):
        gpu = gpus[index % len(gpus)]
        dataset_key = dataset.lower().replace("-", "")
        run_dir = run_root / model / dataset_key / f"h{horizon}" / f"seed{seed}"

        if model == "graphwavenet":
            cwd = baseline_root / "Graph-WaveNet"
            data_dir = (
                package_root
                / "results"
                / "revision_baselines_12step_short"
                / "prepared_data"
                / "graphwavenet"
                / dataset_key
                / f"h{horizon}_622"
            )
            gwn_info = {
                "METR-LA": ("data/sensor_graph/adj_mx.pkl", "207", "gwn_metrla"),
                "PEMS-BAY": ("data/sensor_graph/adj_mx_bay.pkl", "325", "gwn_pemsbay"),
                "PEMS03": ("data/sensor_graph/adj_mx_pems03.pkl", "358", "gwn_pems03"),
                "PEMS07": ("data/sensor_graph/adj_mx_pems07.pkl", "883", "gwn_pems07"),
                "PEMS08": ("data/sensor_graph/adj_mx_pems08.pkl", "170", "gwn_pems08"),
            }
            adjdata, num_nodes, save_prefix = gwn_info[dataset]
            gwn_batch_size = args.gwn_batch_size
            if gwn_batch_size <= 0:
                gwn_batch_size = 16 if dataset == "PEMS07" else 64
            command = [
                python,
                "-u",
                "train.py",
                "--device",
                "cuda:0",
                "--data",
                str(data_dir),
                "--adjdata",
                adjdata,
                "--adjtype",
                "doubletransition",
                "--gcn_bool",
                "--addaptadj",
                "--randomadj",
                "--seq_length",
                str(horizon),
                "--nhid",
                str(args.gwn_nhid),
                "--in_dim",
                "2",
                "--num_nodes",
                num_nodes,
                "--batch_size",
                str(gwn_batch_size),
                "--learning_rate",
                "0.001",
                "--dropout",
                "0.3",
                "--weight_decay",
                "0.0001",
                "--epochs",
                str(epochs),
                "--print_every",
                str(args.gwn_print_every),
                "--save",
                str(run_dir / "checkpoints" / save_prefix),
                "--expid",
                str(seed),
            ]
            if args.copy_numpy_artifacts:
                command.extend(["--artifact_dir", str(run_dir)])
            if args.gwn_stream_raw:
                stream_sources = {
                    "METR-LA": root / "Datasets" / "METR-LA" / "metr-la.h5",
                    "PEMS-BAY": root / "Datasets" / "PEMS-BAY" / "pems-bay.h5",
                    "PEMS03": root / "Datasets" / "PEMS03" / "PEMS03.npz",
                    "PEMS07": root / "Datasets" / "PEMS07" / "PEMS07.npz",
                    "PEMS08": root / "Datasets" / "PEMS08" / "PEMS08.npz",
                }
                command.extend(
                    [
                        "--stream_source",
                        str(stream_sources[dataset]),
                        "--stream_seq_length_x",
                        "12",
                        "--stream_train_ratio",
                        "0.6",
                        "--stream_val_ratio",
                        "0.2",
                        "--stream_test_ratio",
                        "0.2",
                    ]
                )
            if args.gwn_eval_checkpoint:
                command.extend(["--eval_checkpoint", str(args.gwn_eval_checkpoint)])
                if args.gwn_eval_val_loss:
                    command.extend(["--eval_val_loss", str(args.gwn_eval_val_loss)])
        else:
            model_dir_name = "DMFGCRN" if model == "dmfgcrn" else "MTEGCRN"
            cwd = baseline_root / model_dir_name
            recurrent_batch_size = args.recurrent_batch_size
            if recurrent_batch_size <= 0:
                recurrent_batch_size = {
                    "PEMSD3": 40,
                    "PEMSD7": 4,
                }.get(internal_dataset, 64)
            command = [
                python,
                "-u",
                "run_single.py",
                "--dataset",
                internal_dataset,
                "--model",
                model_dir_name,
                "--device",
                "cuda:0",
                "--seed",
                str(seed),
                "--epochs",
                str(epochs),
                "--horizon",
                str(horizon),
                "--val_ratio",
                "0.2",
                "--test_ratio",
                "0.2",
                "--loss_func",
                "mae",
                "--batch_size",
                str(recurrent_batch_size),
                "--early_stop",
                "False",
            ]

        jobs.append(
            Job(
                name=f"{model}_{dataset_key}_h{horizon}_seed{seed}",
                model=model,
                dataset=dataset,
                internal_dataset=internal_dataset,
                horizon=horizon,
                seed=seed,
                gpu=gpu,
                epochs=epochs,
                cwd=cwd,
                run_dir=run_dir,
                command=command,
            )
        )
    return jobs


def launch(args: argparse.Namespace) -> int:
    jobs = build_jobs(args)
    run_root = jobs[0].run_dir.parents[3]
    run_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for job in jobs:
            print(job.gpu, job.name, " ".join(job.command))
        return 0

    processes: list[tuple[Job, subprocess.Popen]] = []
    job_rows: list[dict[str, str]] = []
    for job in jobs:
        job.run_dir.mkdir(parents=True, exist_ok=True)
        (job.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        with (job.run_dir / "command.json").open("w", encoding="utf-8") as fh:
            json.dump({**asdict(job), "cwd": str(job.cwd), "run_dir": str(job.run_dir)}, fh, indent=2, default=str)
        log_fh = (job.run_dir / "run.log").open("w", encoding="utf-8")
        log_fh.write(
            f"[short12] start={time.strftime('%F %T')} model={job.model} "
            f"dataset={job.dataset} horizon={job.horizon} gpu={job.gpu}\n"
        )
        log_fh.write("[short12] command=" + " ".join(job.command) + "\n")
        log_fh.flush()
        proc = subprocess.Popen(
            job.command,
            cwd=job.cwd,
            env=job_env(job.gpu),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((job, proc))
        job_rows.append(
            {
                "name": job.name,
                "model": job.model,
                "dataset": job.dataset,
                "gpu": str(job.gpu),
                "epochs": str(job.epochs),
                "pid": str(proc.pid),
                "status": "running",
                "run_dir": str(job.run_dir),
                "log_path": str(job.run_dir / "run.log"),
            }
        )
    write_csv(run_root / "jobs.csv", job_rows)

    result_rows: list[dict[str, str]] = []
    for job, proc in processes:
        start = time.time()
        returncode = proc.wait()
        status = "completed" if returncode == 0 else "failed"
        log_path = job.run_dir / "run.log"
        with log_path.open("a", encoding="utf-8") as log_fh:
            log_fh.write(
            f"\n[short12] end={time.strftime('%F %T')} "
                f"status={status} returncode={returncode}\n"
            )
        copy_artifacts(job, copy_numpy_artifacts=args.copy_numpy_artifacts)
        parsed = parse_log(job, log_path)
        row = {
            "name": job.name,
            "model": job.model,
            "dataset": job.dataset,
            "internal_dataset": job.internal_dataset,
            "horizon": str(job.horizon),
            "seed": str(job.seed),
            "gpu": str(job.gpu),
            "epochs": str(job.epochs),
            "status": status,
            "returncode": str(returncode),
            "elapsed_wait_sec": f"{time.time() - start:.1f}",
            "run_dir": str(job.run_dir),
            "log_path": str(log_path),
            **parsed,
        }
        result_rows.append(row)
        write_csv(job.run_dir / "metrics.csv", [row])
        write_csv(run_root / "summary.csv", result_rows)
    return 0 if all(row["status"] == "completed" for row in result_rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--run_id", type=str, default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--models", type=str, default="graphwavenet,dmfgcrn,mtegcrn")
    parser.add_argument("--datasets", type=str, default="METR-LA,PEMS-BAY,PEMS08")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gwn_epochs", type=int, default=10)
    parser.add_argument("--gwn_print_every", type=int, default=100)
    parser.add_argument("--dmfg_epochs", type=int, default=3)
    parser.add_argument("--mteg_epochs", type=int, default=10)
    parser.add_argument("--gwn_batch_size", type=int, default=64)
    parser.add_argument("--gwn_nhid", type=int, default=32)
    parser.add_argument("--gwn_stream_raw", action="store_true")
    parser.add_argument("--gwn_eval_checkpoint", type=Path)
    parser.add_argument("--gwn_eval_val_loss", type=str, default="")
    parser.add_argument("--recurrent_batch_size", type=int, default=64)
    parser.add_argument(
        "--copy_numpy_artifacts",
        action="store_true",
        help="copy/save full pred.npy and true.npy artifacts; disabled by default to avoid multi-GB outputs",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
