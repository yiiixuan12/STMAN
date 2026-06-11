#!/usr/bin/env python
"""Export Graph WaveNet pred/true arrays from a finished run directory."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch


GWN_DIR = Path(__file__).resolve().parent / "Graph-WaveNet"
sys.path.insert(0, str(GWN_DIR))

import util  # noqa: E402
from model import gwnet  # noqa: E402


def parse_command_json(path: Path) -> dict[str, str | bool]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    command = payload.get("command", [])
    parsed: dict[str, str | bool] = {}
    index = 0
    while index < len(command):
        item = command[index]
        if isinstance(item, str) and item.startswith("--"):
            key = item[2:]
            if index + 1 < len(command) and not str(command[index + 1]).startswith("--"):
                parsed[key] = str(command[index + 1])
                index += 2
            else:
                parsed[key] = True
                index += 1
        else:
            index += 1
    return parsed


def find_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No Graph WaveNet checkpoint found under {run_dir / 'checkpoints'}")
    best_files = [path for path in checkpoints if "_best_" in path.name]
    if best_files:
        return sorted(best_files, key=lambda p: p.stat().st_mtime)[-1]

    def val_loss(path: Path) -> tuple[float, float]:
        match = re.search(r"_epoch_\d+_([0-9.]+)\.pth$", path.name)
        loss = float(match.group(1)) if match else float("inf")
        return loss, path.stat().st_mtime

    return min(checkpoints, key=val_loss)


def masked_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    mask = np.abs(true) > 1e-8
    if not np.any(mask):
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
    diff = pred - true
    mae = float(np.mean(np.abs(diff[mask])))
    rmse = float(np.sqrt(np.mean(diff[mask] ** 2)))
    mape = float(np.mean(np.abs(diff[mask] / np.maximum(np.abs(true[mask]), 1e-5))))
    return {"mae": mae, "rmse": rmse, "mape": mape}


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


def export_predictions(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir.resolve()
    command_args = parse_command_json(run_dir / "command.json")
    checkpoint = args.checkpoint.resolve() if args.checkpoint else find_checkpoint(run_dir)
    output_dir = (args.output_dir or run_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(str(args.data or command_args.get("data")))
    adjdata_path = Path(str(args.adjdata or command_args.get("adjdata")))
    if not data_path.is_absolute():
        data_path = GWN_DIR / data_path
    if not adjdata_path.is_absolute():
        adjdata_path = GWN_DIR / adjdata_path
    data = str(data_path)
    adjdata = str(adjdata_path)
    adjtype = str(args.adjtype or command_args.get("adjtype", "doubletransition"))
    seq_length = int(args.seq_length or command_args.get("seq_length", 12))
    num_nodes = int(args.num_nodes or command_args.get("num_nodes"))
    in_dim = int(args.in_dim or command_args.get("in_dim", 2))
    nhid = int(args.nhid or command_args.get("nhid", 32))
    batch_size = int(args.batch_size or command_args.get("batch_size", 64))
    dropout = float(args.dropout if args.dropout is not None else command_args.get("dropout", 0.3))
    device = torch.device(args.device)

    _, _, adj_mx = util.load_adj(adjdata, adjtype)
    supports = [torch.tensor(item).to(device) for item in adj_mx]
    adjinit = None if bool(command_args.get("randomadj", args.randomadj)) else supports[0]
    if bool(command_args.get("aptonly", args.aptonly)):
        supports = None

    model = gwnet(
        device,
        num_nodes,
        dropout,
        supports=supports,
        gcn_bool=bool(command_args.get("gcn_bool", args.gcn_bool)),
        addaptadj=bool(command_args.get("addaptadj", args.addaptadj)),
        aptinit=adjinit,
        in_dim=in_dim,
        out_dim=seq_length,
        residual_channels=nhid,
        dilation_channels=nhid,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    dataloader = util.load_dataset(data, batch_size, batch_size, batch_size)
    scaler = dataloader["scaler"]
    realy = torch.tensor(dataloader["y_test"], dtype=torch.float32).to(device).transpose(1, 3)[:, 0, :, :]
    outputs = []
    with torch.no_grad():
        for x, _ in dataloader["test_loader"].get_iterator():
            testx = torch.tensor(x, dtype=torch.float32).to(device).transpose(1, 3)
            outputs.append(model(testx).transpose(1, 3).squeeze())
    yhat = torch.cat(outputs, dim=0)[: realy.size(0), ...]

    pred = scaler.inverse_transform(yhat).transpose(1, 2).unsqueeze(-1).cpu().numpy()
    true = realy.transpose(1, 2).unsqueeze(-1).cpu().numpy()
    np.save(output_dir / "pred.npy", pred)
    np.save(output_dir / "true.npy", true)

    rows = []
    for index in range(pred.shape[1]):
        metrics = masked_metrics(pred[:, index, :, :], true[:, index, :, :])
        rows.append({"horizon": index + 1, **{k: f"{v:.6f}" for k, v in metrics.items()}})
    avg = masked_metrics(pred, true)
    rows.append({"horizon": "average", **{k: f"{v:.6f}" for k, v in avg.items()}})
    write_csv(output_dir / "per_horizon_metrics.csv", rows)

    meta = {
        "model": "graphwavenet",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "data": data,
        "adjdata": adjdata,
        "seq_length": seq_length,
        "num_nodes": num_nodes,
        "shape_pred": list(pred.shape),
        "shape_true": list(true.shape),
    }
    (output_dir / "prediction_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "checkpoint": str(checkpoint), "mae": avg["mae"]}, indent=2))
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--adjdata", type=Path)
    parser.add_argument("--adjtype", type=str)
    parser.add_argument("--seq_length", type=int)
    parser.add_argument("--num_nodes", type=int)
    parser.add_argument("--in_dim", type=int)
    parser.add_argument("--nhid", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--gcn_bool", action="store_true")
    parser.add_argument("--addaptadj", action="store_true")
    parser.add_argument("--randomadj", action="store_true")
    parser.add_argument("--aptonly", action="store_true")
    args = parser.parse_args()
    export_predictions(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
