#!/usr/bin/env python
"""Merge PDFormer/LSTTN unified-MAPE audit CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASET_ORDER = {"METR-LA": 0, "PEMS-BAY": 1, "PEMS03": 2, "PEMS07": 3, "PEMS08": 4}
MODEL_ORDER = {"LSTTN": 0, "PDFormer": 1, "lstnn": 0, "pdformer": 1}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sort_key(row: dict[str, str]) -> tuple[int, int, int]:
    model = row.get("model") or row.get("model_key") or ""
    dataset = row.get("dataset") or ""
    try:
        horizon = int(row.get("horizon", "999999"))
    except ValueError:
        horizon = 999999
    return horizon, DATASET_ORDER.get(dataset, 99), MODEL_ORDER.get(model, 99)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_key: dict[tuple[str, str, int], dict[str, str]] = {}
    for path in args.inputs:
        for row in read_csv(path):
            if not row.get("model") or not row.get("dataset") or not row.get("horizon"):
                continue
            try:
                horizon = int(row["horizon"])
            except ValueError:
                continue
            key = (row.get("model_key") or row["model"], row["dataset"], horizon)
            by_key[key] = row

    rows = sorted(by_key.values(), key=sort_key)
    write_csv(args.output, rows)
    usable = sum(1 for row in rows if row.get("mape_for_table") not in ("", "-", None))
    unstable = sum(1 for row in rows if row.get("mape_for_table") == "-")
    summary = {
        "output": str(args.output),
        "rows": len(rows),
        "usable_mape_for_table": usable,
        "unstable_or_pending_mape": unstable,
        "inputs": [str(path) for path in args.inputs],
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
