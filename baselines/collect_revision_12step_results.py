#!/usr/bin/env python
"""Collect auditable 12-step baseline run logs and prediction artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def latest_run_root(package_root: Path) -> Path:
    parent = package_root / "results" / "revision_baselines_12step_short"
    candidates = sorted(path for path in parent.glob("*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run roots under {parent}")
    return candidates[-1]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_average(block: str) -> tuple[str, str, str]:
    match = re.search(r"Average Horizon, MAE: ([0-9.]+), RMSE: ([0-9.]+), MAPE: ([0-9.]+)", block)
    if match:
        return match.group(1), match.group(2), match.group(3)
    match = re.search(r"On average over \d+ horizons, Test MAE: ([0-9.]+), Test MAPE: ([0-9.]+), Test RMSE: ([0-9.]+)", block)
    if match:
        return match.group(1), match.group(3), match.group(2)
    return "", "", ""


def selected_block(model: str, text: str) -> str:
    if model == "graphwavenet":
        return text
    if "=== Best validation model results ===" in text:
        block = text.split("=== Best validation model results ===", 1)[1]
        return block.split("=== Best test model results ===", 1)[0]
    if "This is best_test_model" in text:
        return text.split("This is best_test_model", 1)[0]
    return text


def parse_log(model: str, log_path: Path) -> dict[str, str]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    row: dict[str, str] = {}
    end_match = re.search(r"\[(?:short12|smoke)\] end=.* status=(\w+) returncode=([0-9-]+)", text)
    if end_match:
        row["status"] = end_match.group(1)
        row["returncode"] = end_match.group(2)
    elif "Traceback" in text:
        row["status"] = "failed_or_running_with_traceback"
    else:
        row["status"] = "running_or_incomplete"

    exp_match = re.search(r"Log directory: (.+)", text)
    if exp_match:
        row["experiment_dir"] = exp_match.group(1).strip()
    exp_match = re.search(r"Experiment log path in: (.+)", text)
    if exp_match and "experiment_dir" not in row:
        row["experiment_dir"] = exp_match.group(1).strip()

    val_losses = [float(item) for item in re.findall(r"Val Epoch \d+: average Loss: ([0-9.]+)", text)]
    if val_losses:
        row["val_mae_min"] = f"{min(val_losses):.6f}"
    if model == "graphwavenet":
        val_match = re.search(r"The valid loss on best model is ([0-9.]+)", text)
        if val_match:
            row["val_mae_best"] = val_match.group(1)

    block = selected_block(model, text)
    mae, rmse, mape = extract_average(block)
    row["test_mae_best_val"] = mae
    row["test_rmse_best_val"] = rmse
    row["test_mape_best_val"] = mape
    return row


def parse_horizons(model: str, log_path: Path) -> list[dict[str, object]]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    block = selected_block(model, text)
    rows: list[dict[str, object]] = []
    if model == "graphwavenet":
        pattern = r"horizon (\d+), Test MAE: ([0-9.]+), Test MAPE: ([0-9.]+), Test RMSE: ([0-9.]+)"
        for horizon, mae, mape, rmse in re.findall(pattern, block):
            rows.append({"horizon": int(horizon), "mae": mae, "rmse": rmse, "mape": mape})
    else:
        pattern = r"Horizon\s+(\d+), MAE: ([0-9.]+), RMSE: ([0-9.]+), MAPE: ([0-9.]+)"
        for horizon, mae, rmse, mape in re.findall(pattern, block):
            rows.append({"horizon": int(horizon), "mae": mae, "rmse": rmse, "mape": mape})
    return rows


def canonical_arrays(pred_path: Path, true_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pred = np.load(pred_path)
    true = np.load(true_path)
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    if true.ndim == 4 and true.shape[-1] == 1:
        true = true[..., 0]
    if pred.ndim != 3 or true.ndim != 3:
        raise ValueError(f"Expected 3D/4D prediction arrays, got {pred.shape} and {true.shape}")
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/true shape mismatch: {pred.shape} vs {true.shape}")
    return pred, true


def masked_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    mask = np.abs(true) > 1e-8
    if not np.any(mask):
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
    diff = pred - true
    return {
        "mae": float(np.mean(np.abs(diff[mask]))),
        "rmse": float(np.sqrt(np.mean(diff[mask] ** 2))),
        "mape": float(np.mean(np.abs(diff[mask] / np.maximum(np.abs(true[mask]), 1e-5)))),
    }


def metrics_from_artifacts(run_dir: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    pred_path = run_dir / "pred.npy"
    true_path = run_dir / "true.npy"
    if not pred_path.exists() or not true_path.exists():
        return {"prediction_artifact": "missing"}, []
    try:
        pred, true = canonical_arrays(pred_path, true_path)
        avg = masked_metrics(pred, true)
        rows = []
        for horizon in range(pred.shape[1]):
            item = masked_metrics(pred[:, horizon, :], true[:, horizon, :])
            rows.append({"horizon": horizon + 1, **{key: f"{value:.6f}" for key, value in item.items()}})
        rows.append({"horizon": "average", **{key: f"{value:.6f}" for key, value in avg.items()}})
        return {
            "prediction_artifact": "present",
            "artifact_shape": "x".join(str(item) for item in pred.shape),
            "artifact_mae": f"{avg['mae']:.6f}",
            "artifact_rmse": f"{avg['rmse']:.6f}",
            "artifact_mape": f"{avg['mape']:.6f}",
            "pred_path": str(pred_path),
            "true_path": str(true_path),
        }, rows
    except Exception as exc:
        return {"prediction_artifact": "error", "artifact_error": str(exc)}, []


def infer_from_path(run_root: Path, run_dir: Path) -> dict[str, str]:
    rel = run_dir.relative_to(run_root).parts
    out = {"model": rel[0] if len(rel) > 0 else "", "dataset_key": rel[1] if len(rel) > 1 else ""}
    if len(rel) > 2 and rel[2].startswith("h"):
        out["horizon"] = rel[2][1:]
    if len(rel) > 3 and rel[3].startswith("seed"):
        out["seed"] = rel[3][4:]
    return out


def collect(run_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    log_paths = sorted(set(run_root.glob("*/*/h*/seed*/run.log")))
    for log_path in log_paths:
        run_dir = log_path.parent
        info = infer_from_path(run_root, run_dir)
        command = read_json(run_dir / "command.json")
        model = str(command.get("model") or info.get("model", ""))
        dataset = str(command.get("dataset") or info.get("dataset_key", ""))
        parsed = parse_log(model, log_path)
        artifact_row, artifact_horizons = metrics_from_artifacts(run_dir)
        row: dict[str, object] = {
            "name": command.get("name", run_dir.name),
            "model": model,
            "dataset": dataset,
            "internal_dataset": command.get("internal_dataset", ""),
            "horizon": command.get("horizon", info.get("horizon", "")),
            "seed": command.get("seed", info.get("seed", "")),
            "gpu": command.get("gpu", ""),
            "epochs": command.get("epochs", ""),
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            **parsed,
            **artifact_row,
        }
        summary_rows.append(row)
        if artifact_row.get("prediction_artifact") == "present":
            manifest_rows.append({
                "model": model,
                "dataset": dataset,
                "internal_dataset": command.get("internal_dataset", ""),
                "horizon": command.get("horizon", info.get("horizon", "")),
                "seed": command.get("seed", info.get("seed", "")),
                "run_dir": str(run_dir),
                "pred_path": artifact_row["pred_path"],
                "true_path": artifact_row["true_path"],
                "selection": "best_val",
                "source": "baseline_run",
            })
        horizon_rows = artifact_horizons or parse_horizons(model, log_path)
        if horizon_rows:
            write_csv(run_dir / "per_horizon_metrics.csv", horizon_rows)
    return summary_rows, manifest_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--run_root", type=Path)
    args = parser.parse_args()
    run_root = (args.run_root or latest_run_root(args.package_root)).resolve()
    rows, manifest = collect(run_root)
    write_csv(run_root / "summary_all.csv", rows)
    write_csv(run_root / "prediction_manifest.csv", manifest)
    print(json.dumps({"run_root": str(run_root), "runs": len(rows), "prediction_artifacts": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
