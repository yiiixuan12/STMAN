"""Shared grid and metric helpers for revision audit scripts."""

from __future__ import annotations

from typing import Iterable


DEFAULT_MODELS = ("graphwavenet", "dmfgcrn", "mtegcrn")
DEFAULT_DATASETS = ("METR-LA", "PEMS-BAY", "PEMS03", "PEMS07", "PEMS08")
DEFAULT_HORIZONS = (12, 48, 96, 288, 864, 2016)
UNSUPPORTED_CELLS = {
    ("graphwavenet", "PEMS08", 2016),
    ("dmfgcrn", "PEMS08", 2016),
    ("mtegcrn", "PEMS08", 2016),
}

METRIC_PRIORITY = (
    "test_mae_best_val",
    "test_mae_min_logged",
    "artifact_mae",
    "val_mae_best",
    "val_mae_min",
    "val_mae",
)


def dataset_key(dataset: str) -> str:
    """Return a path-safe dataset key used by baseline output folders."""

    return dataset.lower().replace("-", "").replace("_", "").replace(" ", "")


def normalize_model(model: str) -> str:
    return model.strip().lower()


def normalize_dataset(dataset: str) -> str:
    text = dataset.strip()
    lookup = {dataset_key(name): name for name in DEFAULT_DATASETS}
    return lookup.get(dataset_key(text), text)


def build_revision_cells(
    models: Iterable[str] = DEFAULT_MODELS,
    datasets: Iterable[str] = DEFAULT_DATASETS,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> set[tuple[str, str, int]]:
    """Build valid baseline cells, excluding known unsupported combinations."""

    cells: set[tuple[str, str, int]] = set()
    for model in models:
        model_key = normalize_model(model)
        for dataset in datasets:
            dataset_name = normalize_dataset(dataset)
            for horizon in horizons:
                cell = (model_key, dataset_name, int(horizon))
                if cell in UNSUPPORTED_CELLS:
                    continue
                cells.add(cell)
    return cells


def metric_value(row: dict[str, object]) -> float | None:
    """Return the preferred MAE-like metric from a summary row."""

    for key in METRIC_PRIORITY:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def build_candidate_rows(
    completed: set[tuple[str, str, int]],
    models: Iterable[str] = DEFAULT_MODELS,
    datasets: Iterable[str] = DEFAULT_DATASETS,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    run_prefix: str = "revision",
) -> list[dict[str, object]]:
    """Return missing cells as queue-friendly rows, shortest horizons first."""

    rows: list[dict[str, object]] = []
    horizon_order = {int(horizon): index for index, horizon in enumerate(sorted({int(h) for h in horizons}))}
    for model, dataset, horizon in sorted(
        build_revision_cells(models=models, datasets=datasets, horizons=horizons),
        key=lambda cell: (horizon_order[cell[2]], cell[0], dataset_key(cell[1])),
    ):
        if (model, dataset, horizon) in completed:
            continue
        priority = horizon_order[horizon] * 10000
        priority += {"graphwavenet": 0, "dmfgcrn": 100, "mtegcrn": 200}.get(model, 900)
        priority += {"metrla": 1, "pemsbay": 2, "pems03": 3, "pems08": 4, "pems07": 5}.get(
            dataset_key(dataset),
            50,
        )
        run_id = f"{run_prefix}_h{horizon}_{model}_{dataset_key(dataset)}"
        rows.append(
            {
                "run_id": run_id,
                "priority": priority,
                "model": model,
                "dataset": dataset,
                "horizon": horizon,
                "dataset_key": dataset_key(dataset),
            }
        )
    return rows

