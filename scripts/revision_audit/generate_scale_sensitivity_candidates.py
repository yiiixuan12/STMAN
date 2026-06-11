#!/usr/bin/env python
"""Generate lightweight STMAN scale-sensitivity probe candidates."""

from __future__ import annotations

import argparse
from pathlib import Path


PROBES = [
    {
        "run_id": "stman_scale_pems08_h12_sf05_tf10",
        "priority": 10,
        "gpu_class": "any",
        "dataset": "PEMS08",
        "horizon": 12,
        "tag": "scale_sf05_tf10",
        "epochs": 6,
        "lr": "3e-4",
        "batch_size": 32,
        "sf_scale": 0.5,
        "tf_scale": 1.0,
    },
    {
        "run_id": "stman_scale_pems08_h12_sf10_tf05",
        "priority": 11,
        "gpu_class": "any",
        "dataset": "PEMS08",
        "horizon": 12,
        "tag": "scale_sf10_tf05",
        "epochs": 6,
        "lr": "3e-4",
        "batch_size": 32,
        "sf_scale": 1.0,
        "tf_scale": 0.5,
    },
    {
        "run_id": "stman_scale_pems08_h12_sf20_tf10",
        "priority": 12,
        "gpu_class": "any",
        "dataset": "PEMS08",
        "horizon": 12,
        "tag": "scale_sf20_tf10",
        "epochs": 6,
        "lr": "2e-4",
        "batch_size": 32,
        "sf_scale": 2.0,
        "tf_scale": 1.0,
    },
    {
        "run_id": "stman_probe_pems07_h48_gcn32",
        "priority": 20,
        "gpu_class": "large",
        "dataset": "PEMS07",
        "horizon": 48,
        "tag": "probe_h48_gcn32",
        "epochs": 8,
        "lr": "3e-4",
        "batch_size": 8,
        "sf_scale": 1.0,
        "tf_scale": 1.0,
    },
    {
        "run_id": "stman_probe_pems07_h96_gcn32",
        "priority": 21,
        "gpu_class": "large",
        "dataset": "PEMS07",
        "horizon": 96,
        "tag": "probe_h96_gcn32",
        "epochs": 8,
        "lr": "3e-4",
        "batch_size": 8,
        "sf_scale": 1.0,
        "tf_scale": 1.0,
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    out = args.output or package_root / "results" / "monitors" / "stman_light_probe_candidates_20260430.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(
            "# run_id|priority|enabled|gpu_class|dataset|horizon|tag|epochs|lr|batch_size|sf_scale|tf_scale|note\n"
        )
        for row in PROBES:
            note = "scale_sensitivity" if row["run_id"].startswith("stman_scale") else "pems07_structural_probe"
            fh.write(
                "|".join(
                    [
                        str(row["run_id"]),
                        str(row["priority"]),
                        "1",
                        str(row["gpu_class"]),
                        str(row["dataset"]),
                        str(row["horizon"]),
                        str(row["tag"]),
                        str(row["epochs"]),
                        str(row["lr"]),
                        str(row["batch_size"]),
                        str(row["sf_scale"]),
                        str(row["tf_scale"]),
                        note,
                    ]
                )
                + "\n"
            )
    print(f"wrote={out} candidates={len(PROBES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
