#!/usr/bin/env python
"""Export and re-evaluate Table III PDFormer/LSTTN retrain checkpoints.

The retrain runner originally reported MAPE with zero targets included in the
denominator clamp.  That is not comparable to the zero-masked MAPE used by the
other revision baselines, especially on METR-LA where test labels contain many
zeros.  This script restores each completed checkpoint, streams the test split,
and writes auditable unified metrics.  Full pred/true arrays are only saved when
they are small enough; long-horizon arrays are summarized without creating
multi-GB artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = CODE_ROOT.parents[1]
BASELINE_ROOT = CODE_ROOT / "baseline"
sys.path.insert(0, str(BASELINE_ROOT))

import train_table3_baseline as table3  # noqa: E402
from train import _forward_bridge, train_model  # noqa: E402


def parse_csv_list(text: str | None) -> set[str] | None:
    if not text:
        return None
    return {item.strip().lower() for item in text.split(",") if item.strip()}


def parse_int_list(text: str | None) -> set[int] | None:
    if not text:
        return None
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def result_candidates(search_root: Path, models: set[str] | None, datasets: set[str] | None, horizons: set[int] | None) -> list[dict[str, object]]:
    candidates: dict[tuple[str, str, int], dict[str, object]] = {}
    for result_path in sorted(search_root.glob("**/result.json")):
        try:
            row = read_json(result_path)
            model = table3.normalize_model(str(row.get("model_key") or row.get("model") or ""))
            dataset = table3.normalize_dataset(str(row.get("dataset") or ""))
            horizon = int(row.get("horizon") or 0)
        except Exception:
            continue
        if models and model not in models and str(row.get("model", "")).lower() not in models:
            continue
        if datasets and table3.dataset_key(dataset) not in datasets and dataset.lower() not in datasets:
            continue
        if horizons and horizon not in horizons:
            continue
        if str(row.get("status")) != "completed":
            continue
        run_dir = Path(str(row.get("run_dir") or result_path.parent))
        best_ckpt = Path(str(row.get("best_ckpt") or run_dir / "checkpoints" / "best.pt"))
        if not best_ckpt.exists():
            continue
        key = (model, dataset, horizon)
        mtime = result_path.stat().st_mtime
        existing = candidates.get(key)
        if existing is None or mtime > float(existing.get("_mtime", 0)):
            candidates[key] = {
                **row,
                "model_key": model,
                "dataset": dataset,
                "horizon": horizon,
                "run_dir": str(run_dir),
                "best_ckpt": str(best_ckpt),
                "result_json": str(result_path),
                "_mtime": mtime,
            }
    return sorted(candidates.values(), key=lambda r: (int(r["horizon"]), str(r["model_key"]), table3.dataset_key(str(r["dataset"]))))


def device_from_args(gpu: int) -> torch.device:
    if gpu < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    if os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES"):
        return torch.device("cuda:0")
    return torch.device(f"cuda:{gpu}")


def stable_epoch_log(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing_epoch_log"
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    if "nan" in text or "inf" in text:
        return False, "nan_or_inf_in_epoch_log"
    return True, "finite_epoch_log"


def build_model_without_training(model_key: str, dataset: str, horizon: int, batch_size: int, device: torch.device):
    cfg = table3.model_config(model_key, epochs=20)
    model_kind = str(cfg.pop("model_kind"))
    lr = float(cfg.pop("lr"))
    patience = int(cfg.pop("patience"))
    adj, train_loader, val_loader, test_loader, scaler, n_nodes, in_dim = table3.load_data(
        CODE_ROOT,
        dataset,
        horizon,
        batch_size,
        model_key,
    )
    model = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        adj=adj,
        n_nodes=n_nodes,
        in_dim=in_dim,
        seq_len=12,
        pred_len=horizon,
        out_dim=1,
        scaler_obj=scaler,
        device=device,
        epochs=0,
        lr=lr,
        patience=patience,
        model_kind=model_kind,
        **cfg,
    )
    xb, yb = next(iter(test_loader))
    with torch.no_grad():
        _forward_bridge(model, xb.to(device), yb.to(device), model_kind=model_kind, teacher_forcing_ratio=0.0)
    return model, model_kind, test_loader, scaler, n_nodes


def inverse_with_scaler(tensor: torch.Tensor, scaler) -> np.ndarray:
    return table3.inverse_with_scaler(tensor, scaler).astype(np.float32, copy=False)


def zero_totals(horizon: int) -> dict[str, np.ndarray | float]:
    return {
        "count_all": np.zeros(horizon, dtype=np.float64),
        "abs_all": np.zeros(horizon, dtype=np.float64),
        "sq_all": np.zeros(horizon, dtype=np.float64),
        "pct_all": np.zeros(horizon, dtype=np.float64),
        "count_gt0": np.zeros(horizon, dtype=np.float64),
        "abs_gt0": np.zeros(horizon, dtype=np.float64),
        "sq_gt0": np.zeros(horizon, dtype=np.float64),
        "pct_gt0_max1": np.zeros(horizon, dtype=np.float64),
        "pct_gt0_eps": np.zeros(horizon, dtype=np.float64),
        "zero_count": 0.0,
        "finite_count": 0.0,
    }


def update_totals(totals: dict[str, np.ndarray | float], pred: np.ndarray, true: np.ndarray) -> None:
    pred = pred[..., 0]
    true = true[..., 0]
    diff = pred - true
    abs_err = np.abs(diff)
    sq_err = diff * diff
    finite = np.isfinite(pred) & np.isfinite(true) & np.isfinite(diff)
    gt0 = finite & (np.abs(true) > 1e-4)
    axes = (0, 2)
    totals["count_all"] += finite.sum(axis=axes)
    totals["abs_all"] += np.where(finite, abs_err, 0.0).sum(axis=axes)
    totals["sq_all"] += np.where(finite, sq_err, 0.0).sum(axis=axes)
    totals["pct_all"] += np.where(finite, abs_err / np.maximum(np.abs(true), 1.0), 0.0).sum(axis=axes)
    totals["count_gt0"] += gt0.sum(axis=axes)
    totals["abs_gt0"] += np.where(gt0, abs_err, 0.0).sum(axis=axes)
    totals["sq_gt0"] += np.where(gt0, sq_err, 0.0).sum(axis=axes)
    totals["pct_gt0_max1"] += np.where(gt0, abs_err / np.maximum(np.abs(true), 1.0), 0.0).sum(axis=axes)
    totals["pct_gt0_eps"] += np.where(gt0, abs_err / (np.abs(true) + 0.001), 0.0).sum(axis=axes)
    totals["zero_count"] = float(totals["zero_count"]) + float((finite & (np.abs(true) <= 1e-4)).sum())
    totals["finite_count"] = float(totals["finite_count"]) + float(finite.sum())


def reduce_metric(total: np.ndarray, count: np.ndarray, rmse: bool = False) -> np.ndarray:
    out = np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float64), where=count > 0)
    if rmse:
        out = np.sqrt(out)
    return out


def summarize_totals(totals: dict[str, np.ndarray | float]) -> tuple[dict[str, float], list[dict[str, object]]]:
    count_all = totals["count_all"]
    count_gt0 = totals["count_gt0"]
    assert isinstance(count_all, np.ndarray) and isinstance(count_gt0, np.ndarray)
    mae_all_h = reduce_metric(totals["abs_all"], count_all)
    rmse_all_h = reduce_metric(totals["sq_all"], count_all, rmse=True)
    mape_all_h = reduce_metric(totals["pct_all"], count_all)
    mae_gt0_h = reduce_metric(totals["abs_gt0"], count_gt0)
    rmse_gt0_h = reduce_metric(totals["sq_gt0"], count_gt0, rmse=True)
    mape_gt0_max1_h = reduce_metric(totals["pct_gt0_max1"], count_gt0)
    mape_gt0_eps_h = reduce_metric(totals["pct_gt0_eps"], count_gt0)

    avg = {
        "mae_all": float(np.nansum(totals["abs_all"]) / max(np.nansum(count_all), 1.0)),
        "rmse_all": float(np.sqrt(np.nansum(totals["sq_all"]) / max(np.nansum(count_all), 1.0))),
        "mape_all_zero_included": float(np.nansum(totals["pct_all"]) / max(np.nansum(count_all), 1.0)),
        "mae_gt0": float(np.nansum(totals["abs_gt0"]) / max(np.nansum(count_gt0), 1.0)),
        "rmse_gt0": float(np.sqrt(np.nansum(totals["sq_gt0"]) / max(np.nansum(count_gt0), 1.0))),
        "mape_gt0_max1": float(np.nansum(totals["pct_gt0_max1"]) / max(np.nansum(count_gt0), 1.0)),
        "mape_gt0_true_eps": float(np.nansum(totals["pct_gt0_eps"]) / max(np.nansum(count_gt0), 1.0)),
        "zero_fraction": float(totals["zero_count"]) / max(float(totals["finite_count"]), 1.0),
        "finite_count": float(totals["finite_count"]),
    }
    rows: list[dict[str, object]] = []
    for idx in range(len(count_all)):
        rows.append(
            {
                "horizon_step": idx + 1,
                "count_all": int(count_all[idx]),
                "count_gt0": int(count_gt0[idx]),
                "mae_all": f"{mae_all_h[idx]:.6f}",
                "rmse_all": f"{rmse_all_h[idx]:.6f}",
                "mape_all_zero_included": f"{mape_all_h[idx]:.6f}",
                "mae_gt0": f"{mae_gt0_h[idx]:.6f}",
                "rmse_gt0": f"{rmse_gt0_h[idx]:.6f}",
                "mape_gt0_max1": f"{mape_gt0_max1_h[idx]:.6f}",
                "mape_gt0_true_eps": f"{mape_gt0_eps_h[idx]:.6f}",
            }
        )
    return avg, rows


def should_save_arrays(n_samples: int, horizon: int, n_nodes: int, mode: str, max_elements: int) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return n_samples * horizon * n_nodes <= max_elements


def export_one(row: dict[str, object], args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    model_key = str(row["model_key"])
    dataset = str(row["dataset"])
    horizon = int(row["horizon"])
    run_dir = Path(str(row["run_dir"]))
    best_ckpt = Path(str(row["best_ckpt"]))
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else min(
        int(row.get("batch_size") or table3.default_batch_size(model_key, dataset, horizon)),
        table3.default_batch_size(model_key, dataset, horizon),
    )
    batch_size = max(1, batch_size)
    pred_path = run_dir / "pred.npy"
    true_path = run_dir / "true.npy"
    metrics_json = run_dir / "metrics_unified_gt0.json"
    per_horizon_csv = run_dir / "per_horizon_metrics_unified_gt0.csv"
    stable, stability_reason = stable_epoch_log(run_dir / "epoch_metrics.tsv")

    model, model_kind, test_loader, scaler, n_nodes = build_model_without_training(
        model_key,
        dataset,
        horizon,
        batch_size,
        device,
    )
    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    n_samples = len(test_loader.dataset)
    save_arrays = should_save_arrays(n_samples, horizon, n_nodes, args.save_arrays, int(args.max_saved_elements))
    pred_chunks: list[np.ndarray] = []
    true_chunks: list[np.ndarray] = []
    totals = zero_totals(horizon)
    start = time.time()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yhat = _forward_bridge(model, xb, yb, model_kind=model_kind, teacher_forcing_ratio=0.0)
            step = min(yhat.shape[1], yb.shape[1], horizon)
            pred = inverse_with_scaler(yhat[:, :step], scaler)
            true = inverse_with_scaler(yb[:, :step, :, :1], scaler)
            update_totals(totals, pred, true)
            if save_arrays:
                pred_chunks.append(pred.astype(np.float32, copy=False))
                true_chunks.append(true.astype(np.float32, copy=False))

    avg, per_horizon = summarize_totals(totals)
    if save_arrays:
        np.save(pred_path, np.concatenate(pred_chunks, axis=0))
        np.save(true_path, np.concatenate(true_chunks, axis=0))
    write_csv(per_horizon_csv, per_horizon)
    payload = {
        "model": model_key,
        "dataset": dataset,
        "horizon": horizon,
        "run_dir": str(run_dir),
        "best_ckpt": str(best_ckpt),
        "batch_size_eval": batch_size,
        "save_arrays": save_arrays,
        "pred_path": str(pred_path) if save_arrays else "",
        "true_path": str(true_path) if save_arrays else "",
        "per_horizon_csv": str(per_horizon_csv),
        "metric_policy": "gt0 mask; mape_gt0_max1 uses |true|>1e-4 and denominator max(|true|,1)",
        "training_stable": stable,
        "stability_reason": stability_reason,
        "elapsed_sec": time.time() - start,
        "metrics": avg,
        "raw_result": row,
    }
    metrics_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "model": "LSTTN" if model_key == "lstnn" else "PDFormer",
        "model_key": model_key,
        "dataset": dataset,
        "horizon": horizon,
        "status": "completed",
        "training_stable": stable,
        "stability_reason": stability_reason,
        "mae_raw": row.get("mae", ""),
        "rmse_raw": row.get("rmse", ""),
        "mape_raw_zero_included": row.get("mape", ""),
        "smape_raw": row.get("smape", ""),
        "mae_all_recomputed": f"{avg['mae_all']:.6f}",
        "rmse_all_recomputed": f"{avg['rmse_all']:.6f}",
        "mape_all_zero_included_recomputed": f"{avg['mape_all_zero_included']:.6f}",
        "mae_gt0": f"{avg['mae_gt0']:.6f}",
        "rmse_gt0": f"{avg['rmse_gt0']:.6f}",
        "mape_gt0_max1": f"{avg['mape_gt0_max1']:.6f}",
        "mape_gt0_true_eps": f"{avg['mape_gt0_true_eps']:.6f}",
        "mape_for_table": f"{avg['mape_gt0_max1']:.6f}" if stable else "-",
        "mape_for_table_note": "use unified gt0 metric" if stable else "use '-' until stable rerun",
        "zero_fraction": f"{avg['zero_fraction']:.6f}",
        "finite_count": f"{avg['finite_count']:.0f}",
        "pred_path": str(pred_path) if save_arrays else "",
        "true_path": str(true_path) if save_arrays else "",
        "metrics_json": str(metrics_json),
        "per_horizon_csv": str(per_horizon_csv),
        "run_dir": str(run_dir),
        "best_ckpt": str(best_ckpt),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search_root", type=Path, default=PACKAGE_ROOT / "results" / "revision_baselines_table3_retrain")
    parser.add_argument("--output_csv", type=Path, default=PACKAGE_ROOT / "results" / "revision_audit_20260518" / "table3_pdformer_lsttn_unified_mape_audit.csv")
    parser.add_argument("--models", type=str, default="lstnn,pdformer")
    parser.add_argument("--datasets", type=str)
    parser.add_argument("--horizons", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--save_arrays", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--max_saved_elements", type=int, default=30_000_000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    models = parse_csv_list(args.models)
    datasets = parse_csv_list(args.datasets)
    horizons = parse_int_list(args.horizons)
    rows = result_candidates(args.search_root.resolve(), models, datasets, horizons)
    if args.limit > 0:
        rows = rows[: args.limit]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.output_csv.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "search_root": str(args.search_root.resolve()),
                "output_csv": str(args.output_csv),
                "models": sorted(models or []),
                "datasets": sorted(datasets or []),
                "horizons": sorted(horizons or []),
                "cells": len(rows),
                "started_at": time.strftime("%F %T"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    device = device_from_args(int(args.gpu))
    out_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        print(f"[export] {index}/{len(rows)} {row['model_key']}/{row['dataset']}/h{row['horizon']} -> {row['run_dir']}", flush=True)
        try:
            out = export_one(row, args, device)
            out_rows.append(out)
            write_csv(args.output_csv, out_rows)
            print(
                "[export] done mae_all={mae} mape_raw={raw} mape_gt0={gt0} stable={stable}".format(
                    mae=out["mae_all_recomputed"],
                    raw=out["mape_all_zero_included_recomputed"],
                    gt0=out["mape_gt0_max1"],
                    stable=out["training_stable"],
                ),
                flush=True,
            )
        except Exception as exc:
            failed = {
                "model": row.get("model", row.get("model_key", "")),
                "model_key": row.get("model_key", ""),
                "dataset": row.get("dataset", ""),
                "horizon": row.get("horizon", ""),
                "status": "export_failed",
                "error": repr(exc),
                "run_dir": row.get("run_dir", ""),
                "best_ckpt": row.get("best_ckpt", ""),
            }
            out_rows.append(failed)
            write_csv(args.output_csv, out_rows)
            print(f"[export] failed {failed}", flush=True)
    print(json.dumps({"output_csv": str(args.output_csv), "cells": len(out_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
