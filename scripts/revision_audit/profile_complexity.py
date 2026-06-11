#!/usr/bin/env python
"""Profile parameter count, checkpoint footprint, and log-derived runtime."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean

from revision_audit.grid import dataset_key, normalize_dataset, normalize_model


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


def tensor_state(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(ckpt.get(key), dict):
                return ckpt[key]
    return ckpt if isinstance(ckpt, dict) else {}


def count_state_dict(path: Path) -> tuple[int, int]:
    import torch

    ckpt = torch.load(path, map_location="cpu")
    state = tensor_state(ckpt)
    total = 0
    bytes_ = 0
    for value in state.values():
        if hasattr(value, "numel"):
            total += int(value.numel())
            bytes_ += int(value.numel() * value.element_size())
    return total, bytes_


def parse_baseline_log(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    params = ""
    match = re.search(r"Total params num:\s*([0-9]+)", text)
    if match:
        params = match.group(1)
    epoch_times = [float(x) for x in re.findall(r"Training Time:\s*([0-9.]+)/epoch", text)]
    epoch_times += [float(x) for x in re.findall(r"Training Time:\s*([0-9.]+)\s*secs/epoch", text)]
    epoch_times += [float(x) for x in re.findall(r"Train Epoch \d+:.*?train time:\s*([0-9.]+)\s*s", text)]
    return {
        "param_count": params,
        "mean_epoch_sec": f"{mean(epoch_times):.2f}" if epoch_times else "",
        "epoch_time_samples": len(epoch_times),
    }


def infer_from_run_dir(run_dir: Path) -> dict[str, str]:
    parts = run_dir.parts
    try:
        idx = parts.index("revision_baselines_12step_short")
        model, dataset, horizon, seed = parts[idx + 2 : idx + 6]
    except Exception:
        return {}
    return {
        "model": normalize_model(model),
        "dataset": normalize_dataset(dataset),
        "horizon": horizon.removeprefix("h"),
        "seed": seed.removeprefix("seed"),
    }


def collect_baseline_rows(package_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for log_path in sorted((package_root / "results" / "revision_baselines_12step_short").glob("*/*/*/h*/seed*/run.log")):
        run_dir = log_path.parent
        info = infer_from_run_dir(run_dir)
        if not info:
            continue
        profile = parse_baseline_log(log_path)
        checkpoint_params = ""
        checkpoint_mb = ""
        ckpts = sorted((run_dir / "checkpoints").glob("*.pth"))
        if ckpts:
            try:
                checkpoint_params, bytes_ = count_state_dict(ckpts[-1])
                checkpoint_mb = f"{bytes_ / (1024 * 1024):.3f}"
            except Exception:
                pass
        rows.append(
            {
                **info,
                "source": "baseline_run",
                "param_count": checkpoint_params or profile["param_count"],
                "checkpoint_mb": checkpoint_mb,
                "mean_epoch_sec": profile["mean_epoch_sec"],
                "epoch_time_samples": profile["epoch_time_samples"],
                "log_path": str(log_path),
            }
        )
    return rows


def collect_stman_rows(package_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((package_root / "checkpoint" / "stman").glob("*_best.pt")):
        match = re.match(r"(.+?)_(\d+)_(.+?)_best\.pt$", path.name)
        if not match:
            continue
        try:
            params, bytes_ = count_state_dict(path)
        except Exception:
            continue
        rows.append(
            {
                "model": "stman",
                "dataset": match.group(1),
                "horizon": match.group(2),
                "tag": match.group(3),
                "source": "stman_checkpoint",
                "param_count": params,
                "checkpoint_mb": f"{bytes_ / (1024 * 1024):.3f}",
                "checkpoint": str(path),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    output_dir = (args.output_dir or package_root / "results" / "revision_audit_20260430").resolve()
    rows = collect_baseline_rows(package_root) + collect_stman_rows(package_root)
    out = output_dir / "complexity_profile.csv"
    write_csv(out, rows)
    summary = {
        "csv": str(out),
        "rows": len(rows),
        "baseline_rows": sum(1 for row in rows if row.get("source") == "baseline_run"),
        "stman_rows": sum(1 for row in rows if row.get("source") == "stman_checkpoint"),
        "datasets": sorted({dataset_key(str(row.get("dataset", ""))) for row in rows if row.get("dataset")}),
    }
    (output_dir / "complexity_profile_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

