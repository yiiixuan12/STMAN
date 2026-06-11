#!/usr/bin/env python
"""Generate a dynamic baseline full-grid candidate TSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from revision_audit.grid import (
    DEFAULT_DATASETS,
    DEFAULT_HORIZONS,
    DEFAULT_MODELS,
    build_candidate_rows,
    dataset_key,
    metric_value,
    normalize_dataset,
    normalize_model,
)


def completed_cells(package_root: Path) -> set[tuple[str, str, int]]:
    cells: set[tuple[str, str, int]] = set()
    for summary in sorted((package_root / "results" / "revision_baselines_12step_short").glob("*/summary*.csv")):
        with summary.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                model = normalize_model(str(row.get("model", "")))
                dataset = normalize_dataset(str(row.get("dataset", "")))
                horizon_text = str(row.get("horizon", "")).lower().removeprefix("h")
                try:
                    horizon = int(horizon_text)
                except ValueError:
                    continue
                if metric_value(row) is None:
                    continue
                status = str(row.get("status", "")).lower()
                if status == "completed":
                    cells.add((model, dataset, horizon))
    return cells


def default_epochs(model: str, horizon: int) -> int:
    if model == "dmfgcrn":
        return 10 if horizon <= 96 else 5
    if model == "mtegcrn":
        return 10 if horizon <= 288 else 5
    return 10 if horizon <= 96 else 5


def default_batch(model: str, dataset: str, horizon: int) -> int:
    if dataset == "PEMS07":
        return 8 if model == "graphwavenet" else 4
    if dataset == "PEMS03":
        return 40
    if horizon >= 864 and model != "graphwavenet":
        return 32
    return 64


def py_env(model: str) -> str:
    return "vllm" if model == "graphwavenet" else "torch27"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run_prefix", type=str, default="20260430_fullgrid")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    out = args.output or package_root / "results" / "monitors" / "baseline_fullgrid_candidates_20260430.tsv"
    done = completed_cells(package_root)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    candidates = build_candidate_rows(
        completed=done,
        models=models,
        datasets=DEFAULT_DATASETS,
        horizons=DEFAULT_HORIZONS,
        run_prefix=args.run_prefix,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write("# run_id|priority|enabled|model|dataset|horizon|epochs|batch_size|py_env|note\n")
        for row in candidates:
            model = str(row["model"])
            dataset = str(row["dataset"])
            horizon = int(row["horizon"])
            priority = int(row["priority"]) + {"dmfgcrn": 0, "mtegcrn": 100, "graphwavenet": 500}.get(model, 900)
            note = f"fullgrid_backfill_{dataset_key(dataset)}_h{horizon}"
            fh.write(
                "|".join(
                    [
                        str(row["run_id"]),
                        str(priority),
                        "1",
                        model,
                        dataset,
                        str(horizon),
                        str(default_epochs(model, horizon)),
                        str(default_batch(model, dataset, horizon)),
                        py_env(model),
                        note,
                    ]
                )
                + "\n"
            )
    print(f"wrote={out} completed={len(done)} candidates={len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
