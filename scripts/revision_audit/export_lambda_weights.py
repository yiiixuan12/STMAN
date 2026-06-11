#!/usr/bin/env python
"""Export learned STMAN lambda weights from checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def softplus_values(tensor):
    import torch

    return torch.nn.functional.softplus(tensor.detach().cpu().float()).tolist()


def checkpoint_state(ckpt: dict) -> dict:
    for key in ("model", "model_state_dict", "state_dict"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    return ckpt


def parse_name(path: Path) -> dict[str, str]:
    match = re.match(r"(.+?)_(\d+)_(.+?)_(best|last)\.pt$", path.name)
    if not match:
        return {"dataset": "", "horizon": "", "tag": "", "kind": ""}
    return {
        "dataset": match.group(1),
        "horizon": match.group(2),
        "tag": match.group(3),
        "kind": match.group(4),
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--pattern", type=str, default="*_best.pt")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--max_files", type=int, default=0)
    args = parser.parse_args()

    import torch

    package_root = args.package_root.resolve()
    output_dir = (args.output_dir or package_root / "results" / "revision_audit_20260430").resolve()
    ckpt_dir = package_root / "checkpoint" / "stman"
    paths = sorted(ckpt_dir.glob(args.pattern))
    if args.max_files > 0:
        paths = paths[-args.max_files :]

    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            ckpt = torch.load(path, map_location="cpu")
            if not isinstance(ckpt, dict):
                raise TypeError(f"unexpected checkpoint type: {type(ckpt)!r}")
            state = checkpoint_state(ckpt)
            st = state.get("biattn._lambda_st")
            ts = state.get("biattn._lambda_ts")
            if st is None or ts is None:
                continue
            st_values = softplus_values(st)
            ts_values = softplus_values(ts)
            meta = parse_name(path)
            config = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
            rows.append(
                {
                    **meta,
                    "checkpoint": str(path),
                    "epoch": ckpt.get("epoch", ""),
                    "val_loss": ckpt.get("val_loss", ""),
                    "spatial_mode": config.get("spatial_mode", ""),
                    "direct_head_mode": config.get("direct_head_mode", ""),
                    "lambda_st_adj": f"{st_values[0]:.6f}",
                    "lambda_st_spatial_fractal": f"{st_values[1]:.6f}",
                    "lambda_st_temporal_fractal": f"{st_values[2]:.6f}",
                    "lambda_ts_adj": f"{ts_values[0]:.6f}",
                    "lambda_ts_spatial_fractal": f"{ts_values[1]:.6f}",
                    "lambda_ts_temporal_fractal": f"{ts_values[2]:.6f}",
                }
            )
        except Exception as exc:  # keep the export robust across old checkpoints
            errors.append({"checkpoint": str(path), "error": str(exc)})

    csv_path = output_dir / "lambda_weights.csv"
    write_csv(csv_path, rows)
    manifest = {"checkpoints_seen": len(paths), "exported": len(rows), "errors": errors, "csv": str(csv_path)}
    (output_dir / "lambda_weights_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

