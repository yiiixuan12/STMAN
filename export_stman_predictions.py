#!/usr/bin/env python
"""Export STMAN pred/true arrays for real revision figures."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


CODE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_DIR.parents[1]
sys.path.insert(0, str(CODE_DIR))

from batch_eval import (  # noqa: E402
    compute_eval_metrics,
    detect_model_config,
    inverse_speed_channel,
    split_raw_by_protocol,
)
from model import STFractalTransformer  # noqa: E402
from utils import append_calendar_features, create_data_loaders, load_adjacency_csv  # noqa: E402


def load_frozen_map(path: Path) -> dict[tuple[str, int], str]:
    frozen: dict[tuple[str, int], str] = {}
    if not path.exists():
        return frozen
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                frozen[(parts[0], int(parts[1]))] = parts[2]
            except ValueError:
                continue
    return frozen


def resolve_checkpoint(checkpoint_dir: Path, dataset: str, pred_len: int, tag: str | None) -> tuple[Path, str]:
    if not tag:
        tag = load_frozen_map(checkpoint_dir / "FROZEN.txt").get((dataset, pred_len))
    if tag:
        ckpt = checkpoint_dir / f"{dataset}_{pred_len}_{tag}_best.pt"
        if ckpt.exists():
            return ckpt, tag
    candidates = sorted(
        checkpoint_dir.glob(f"{dataset}_{pred_len}_*_best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No STMAN checkpoint for {dataset} horizon {pred_len} in {checkpoint_dir}")
    ckpt = candidates[0]
    stem = ckpt.name[: -len("_best.pt")]
    prefix = f"{dataset}_{pred_len}_"
    return ckpt, stem[len(prefix):]


def load_raw_data_fallback(dataset: str, add_dow: bool = False) -> np.ndarray:
    data_dir = CODE_DIR / "Datasets" / dataset
    npz_path = data_dir / f"{dataset}.npz"
    if npz_path.exists():
        raw = np.load(npz_path)["data"].astype(np.float32)
    else:
        raw = None
        for h5_name in ["metr-la.h5", "pems-bay.h5"]:
            h5_path = data_dir / h5_name
            if not h5_path.exists():
                continue
            try:
                raw = pd.read_hdf(h5_path).values.astype(np.float32)
                break
            except ImportError:
                raw = None
        if raw is None:
            csv_path = data_dir / f"{dataset}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"No data for {dataset}")
            raw = pd.read_csv(csv_path, header=None).values.astype(np.float32)
    if raw.ndim == 2:
        raw = raw[:, :, np.newaxis]
    return append_calendar_features(raw, steps_per_day=288, add_dow=add_dow)


def masked_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(true) & (np.abs(true) > 1e-4)
    if not np.any(mask):
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
    diff = pred - true
    mae = float(np.mean(np.abs(diff[mask])))
    rmse = float(np.sqrt(np.mean(diff[mask] ** 2)))
    mape = float(np.mean(np.abs(diff[mask]) / np.maximum(np.abs(true[mask]), 1.0)))
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
    os.chdir(CODE_DIR)
    checkpoint_dir = args.checkpoint_dir.resolve()
    ckpt_path, tag = resolve_checkpoint(checkpoint_dir, args.dataset, args.pred_len, args.tag)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else PACKAGE_ROOT
        / "results"
        / "revision_predictions"
        / "stman"
        / args.dataset.lower().replace("-", "")
        / f"h{args.pred_len}"
        / tag
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.gpu < 0 or not torch.cuda.is_available() else f"cuda:{args.gpu}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = detect_model_config(ckpt)
    raw = load_raw_data_fallback(args.dataset, add_dow=False)
    if raw.shape[-1] != mcfg.get("in_dim", raw.shape[-1]):
        needs_dow = (
            mcfg.get("calendar_features") == "tod+dow"
            or mcfg.get("in_dim", 0) == raw.shape[-1] + 1
        )
        if needs_dow:
            raw = load_raw_data_fallback(args.dataset, add_dow=True)
    if raw.shape[-1] != mcfg.get("in_dim", raw.shape[-1]):
        raise ValueError(f"in_dim mismatch: model={mcfg.get('in_dim')} data={raw.shape[-1]}")
    x_train, x_val, x_test = split_raw_by_protocol(raw, args.protocol)
    _, _, test_loader, scaler = create_data_loaders(
        x_train,
        x_val,
        x_test,
        seq_len=mcfg["seq_len"],
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        scaler_type=mcfg["scaler"],
        scale_target_only=mcfg.get("scale_target_only", False),
        target_dim=1,
        verbose=False,
    )

    data_dir = CODE_DIR / "Datasets" / args.dataset
    adjacency = load_adjacency_csv(
        str(data_dir / "adj_matrix.csv"),
        normalize="sym",
        add_self_loops=True,
    ).to(device)
    spatial = torch.tensor(
        np.load(CODE_DIR / f"spatial_fractal_vectors_{args.dataset}.npy"),
        dtype=torch.float32,
    ).to(device)
    temporal = torch.tensor(
        np.load(CODE_DIR / f"train_timefractals_{args.dataset}.npy"),
        dtype=torch.float32,
    ).to(device)

    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model = STFractalTransformer(
        n_nodes=adjacency.shape[0],
        in_dim=raw.shape[-1],
        out_dim=1,
        seq_len=mcfg["seq_len"],
        pred_len=args.pred_len,
        sf_dim=spatial.shape[1],
        tf_dim=temporal.shape[1],
        d_model=mcfg["d_model"],
        n_heads=mcfg["n_heads"],
        num_encoder_layers=mcfg["enc_layers"],
        num_decoder_layers=2,
        dropout=0.0,
        ff_multiplier=mcfg["ff_multiplier"],
        use_direct_pred=mcfg["use_direct_pred"],
        direct_head_mode=mcfg["direct_head_mode"],
        direct_step_refine=mcfg["direct_step_refine"],
        decoder_future_tod=mcfg["decoder_future_tod"],
        gated_residual=mcfg.get("gated_residual", False),
        use_future_time_features=mcfg.get("use_future_time_features", False),
        time_feature_start=mcfg.get("time_feature_start", 1),
        spatial_mode=mcfg["spatial_mode"],
        k_hop=mcfg["k_hop"],
    )
    normalized_state = {key.replace("module.", "").replace("inner.", ""): value for key, value in state.items()}
    try:
        model.load_state_dict(normalized_state, strict=True)
    except RuntimeError:
        model.load_state_dict(normalized_state, strict=False)
    model.to(device).eval()

    all_preds, all_trues, all_last = [], [], []
    with torch.no_grad():
        for index, (xb, yb) in enumerate(test_loader):
            if args.max_batches > 0 and index >= args.max_batches:
                break
            yh, _ = model(xb.to(device), adjacency, spatial, temporal)
            pred_norm = yh[:, :, :, 0].cpu().numpy()
            true_norm = yb[:, :, :, 0].cpu().numpy()
            last_norm = xb[:, -1:, :, 0].cpu().numpy()
            all_preds.append(inverse_speed_channel(pred_norm, scaler))
            all_trues.append(inverse_speed_channel(true_norm, scaler))
            all_last.append(inverse_speed_channel(last_norm, scaler))

    pred = np.concatenate(all_preds, axis=0)[:, :, :, None]
    true = np.concatenate(all_trues, axis=0)[:, :, :, None]
    last_values = np.concatenate(all_last, axis=0)
    np.save(output_dir / "pred.npy", pred)
    np.save(output_dir / "true.npy", true)
    np.save(output_dir / "last_value.npy", last_values)

    metrics = compute_eval_metrics(pred[:, :, :, 0], true[:, :, :, 0], last_values, protocol=args.protocol)
    rows = []
    for horizon in range(pred.shape[1]):
        h_metrics = masked_metrics(pred[:, horizon, :, :], true[:, horizon, :, :])
        rows.append({"horizon": horizon + 1, **{key: f"{value:.6f}" for key, value in h_metrics.items()}})
    avg = masked_metrics(pred, true)
    rows.append({"horizon": "average", **{key: f"{value:.6f}" for key, value in avg.items()}})
    write_csv(output_dir / "per_horizon_metrics.csv", rows)

    meta = {
        "model": "stman",
        "dataset": args.dataset,
        "horizon": args.pred_len,
        "tag": tag,
        "protocol": args.protocol,
        "checkpoint": str(ckpt_path),
        "output_dir": str(output_dir),
        "shape_pred": list(pred.shape),
        "shape_true": list(true.shape),
        "metrics": {k: v for k, v in metrics.items() if k != "step_mae"},
        "config": mcfg,
    }
    (output_dir / "prediction_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "checkpoint": str(ckpt_path), "mae": avg["mae"]}, indent=2))
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--tag")
    parser.add_argument("--protocol", choices=["current", "paper"], default="current")
    parser.add_argument("--checkpoint_dir", type=Path, default=PACKAGE_ROOT / "checkpoint" / "stman")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=0)
    args = parser.parse_args()
    export_predictions(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
