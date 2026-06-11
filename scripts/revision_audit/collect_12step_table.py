#!/usr/bin/env python
"""Collect the auditable 12-step baseline table across all short-run folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from revision_audit.grid import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    build_candidate_rows,
    dataset_key,
    metric_value,
    normalize_dataset,
    normalize_model,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def row_horizon(row: dict[str, str]) -> int | None:
    value = str(row.get("horizon", "")).strip().lower().removeprefix("h")
    try:
        return int(value)
    except ValueError:
        return None


def status_rank(row: dict[str, str]) -> tuple[int, int, float]:
    status = str(row.get("status", "")).lower()
    completed = 1 if status == "completed" and str(row.get("returncode", "0")) in ("", "0") else 0
    artifact = 1 if row.get("prediction_artifact") == "present" or row.get("pred_path") else 0
    metric = metric_value(row)
    return completed, artifact, -metric if metric is not None else float("-inf")


def normalize_summary_row(row: dict[str, str], source_summary: Path) -> dict[str, object] | None:
    model = normalize_model(str(row.get("model", "")))
    if model not in DEFAULT_MODELS:
        return None
    dataset = normalize_dataset(str(row.get("dataset", "")))
    if dataset_key(dataset) not in {dataset_key(item) for item in DEFAULT_DATASETS}:
        return None
    horizon = row_horizon(row)
    if horizon != 12:
        return None
    mae = metric_value(row)
    if mae is None:
        return None
    out: dict[str, object] = {
        "model": model,
        "dataset": dataset,
        "horizon": horizon,
        "mae": f"{mae:.4f}",
        "rmse": row.get("test_rmse_best_val") or row.get("artifact_rmse") or "",
        "mape": row.get("test_mape_best_val") or row.get("artifact_mape") or "",
        "val_mae": row.get("val_mae_best") or row.get("val_mae_min") or "",
        "status": row.get("status", ""),
        "returncode": row.get("returncode", ""),
        "epochs": row.get("epochs", ""),
        "seed": row.get("seed", ""),
        "run_dir": row.get("run_dir", ""),
        "log_path": row.get("log_path", ""),
        "pred_path": row.get("pred_path", ""),
        "true_path": row.get("true_path", ""),
        "source_summary": str(source_summary),
        "metric_source": metric_source(row),
    }
    return out


def metric_source(row: dict[str, str]) -> str:
    for key in ("test_mae_best_val", "test_mae_min_logged", "artifact_mae", "val_mae_best", "val_mae_min"):
        if row.get(key) not in (None, ""):
            return key
    return ""


def collect_rows(run_root: Path) -> list[dict[str, object]]:
    best: dict[tuple[str, str, int], dict[str, object]] = {}
    best_rank: dict[tuple[str, str, int], tuple[int, int, float]] = {}
    for summary in sorted(run_root.glob("*/summary*.csv")):
        for raw in read_csv(summary):
            row = normalize_summary_row(raw, summary)
            if row is None:
                continue
            key = (str(row["model"]), str(row["dataset"]), int(row["horizon"]))
            rank = status_rank(raw)
            if key not in best or rank > best_rank[key]:
                best[key] = row
                best_rank[key] = rank

    rows: list[dict[str, object]] = []
    for model in DEFAULT_MODELS:
        for dataset in DEFAULT_DATASETS:
            key = (model, dataset, 12)
            if key in best:
                rows.append({**best[key], "coverage": "complete"})
            else:
                rows.append({"model": model, "dataset": dataset, "horizon": 12, "coverage": "missing"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    run_root = package_root / "results" / "revision_baselines_12step_short"
    output_dir = (args.output_dir or package_root / "results" / "revision_audit_20260430").resolve()
    rows = collect_rows(run_root)
    completed = {
        (str(row["model"]), str(row["dataset"]), int(row["horizon"]))
        for row in rows
        if row.get("coverage") == "complete"
    }
    missing_h12 = build_candidate_rows(
        completed=completed,
        models=DEFAULT_MODELS,
        datasets=DEFAULT_DATASETS,
        horizons=[12],
        run_prefix="missing_h12",
    )
    out_path = output_dir / "baseline_12step_main_table.csv"
    write_csv(out_path, rows)
    summary = {
        "table": str(out_path),
        "completed_h12_cells": len(completed),
        "missing_h12_cells": len(missing_h12),
        "nominal_full_grid_cells": 90,
        "valid_full_grid_cells": 87,
        "remaining_valid_full_grid_after_h12": 87 - len(completed),
        "remaining_nominal_full_grid_after_h12": 90 - len(completed),
    }
    (output_dir / "baseline_12step_main_table_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

