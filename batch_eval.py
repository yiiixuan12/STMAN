#!/usr/bin/env python
"""Batch evaluation: for each dataset×horizon, find best checkpoint and evaluate on test set."""
import os, sys, glob, re, time, subprocess, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import append_calendar_features, load_adjacency_csv, create_data_loaders
from model import STFractalTransformer

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))

TARGETS = {
    ("METR-LA", 12): 3.28, ("METR-LA", 48): 5.19, ("METR-LA", 96): 7.80,
    ("METR-LA", 288): 9.44, ("METR-LA", 864): 10.50, ("METR-LA", 2016): 11.13,
    ("PEMS-BAY", 12): 1.86, ("PEMS-BAY", 48): 2.59, ("PEMS-BAY", 96): 2.78,
    ("PEMS-BAY", 288): 2.98, ("PEMS-BAY", 864): 3.17, ("PEMS-BAY", 2016): 3.39,
    ("PEMS03", 12): 14.42, ("PEMS03", 48): 15.36, ("PEMS03", 96): 17.04,
    ("PEMS03", 288): 18.36, ("PEMS03", 864): 20.09, ("PEMS03", 2016): 22.92,
    ("PEMS07", 12): 22.32, ("PEMS07", 48): 24.07, ("PEMS07", 96): 25.90,
    ("PEMS07", 288): 27.63, ("PEMS07", 864): 29.73, ("PEMS07", 2016): 31.99,
    ("PEMS08", 12): 13.75, ("PEMS08", 48): 17.31, ("PEMS08", 96): 21.76,
    ("PEMS08", 288): 22.37, ("PEMS08", 864): 24.46, ("PEMS08", 2016): 26.23,
}

PROTOCOL_SPLITS = {
    "current": (0.6, 0.2),
    "paper": (0.7, 0.1),
}

def classify_status(mae, target, diversity, vs_persistence):
    gap = (mae - target) / target * 100
    if diversity < 0.2:
        status = "COLLAPSED"
    elif mae <= target and diversity > 0.5 and vs_persistence < 0.95:
        status = "TRUE_REACHED"
    elif mae <= target:
        status = "BORDERLINE"
    elif gap < 5:
        status = "CLOSE"
    elif gap < 20:
        status = "MODERATE"
    else:
        status = "FAR"
    return status, gap

def load_raw_data(ds, add_dow=False):
    data_dir = f"Datasets/{ds}"
    npz_path = os.path.join(data_dir, f"{ds}.npz")
    h5_candidates = [os.path.join(data_dir, "metr-la.h5"), os.path.join(data_dir, "pems-bay.h5")]
    csv_path = os.path.join(data_dir, f"{ds}.csv")

    if os.path.exists(npz_path):
        raw = np.load(npz_path)["data"].astype(np.float32)
    else:
        raw = None
        for h5 in h5_candidates:
            if os.path.exists(h5):
                try:
                    raw = pd.read_hdf(h5).values.astype(np.float32)
                    break
                except ImportError:
                    raw = None
        else:
            raw = None
        if raw is None:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"No data for {ds}")
            raw = pd.read_csv(csv_path, header=None).values.astype(np.float32)

    if raw.ndim == 2:
        raw = raw[:, :, np.newaxis]
    return append_calendar_features(raw, steps_per_day=288, add_dow=add_dow)

def resolve_max_batches(protocol, max_batches):
    if max_batches is not None:
        return max_batches
    return 0 if protocol == "paper" else 30

def select_eval_loader(split, train_loader, val_loader, test_loader):
    if split == "val":
        return val_loader
    if split == "test":
        return test_loader
    raise ValueError(f"Unknown eval split: {split}")

def split_raw_by_protocol(raw, protocol):
    if protocol not in PROTOCOL_SPLITS:
        raise ValueError(f"Unknown protocol: {protocol}")
    train_ratio, val_ratio = PROTOCOL_SPLITS[protocol]
    total = raw.shape[0]
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    return raw[:n_train], raw[n_train:n_train+n_val], raw[n_train+n_val:]

def compute_eval_metrics(preds, trues, last_vals, protocol="current"):
    if protocol == "paper":
        mask = trues > 0
    else:
        mask = np.abs(trues) > 1e-4

    if mask.sum() == 0:
        return {"error": "no valid data"}

    abs_err = np.abs(preds - trues)
    sq_err = (preds - trues) ** 2
    rel_err = abs_err / np.maximum(np.abs(trues), 1.0)

    mae = float(np.mean(abs_err[mask]))
    rmse = float(np.sqrt(np.mean(sq_err[mask])))
    mape = float(np.mean(rel_err[mask])) * 100

    pred_std_all = float(np.mean(np.std(preds, axis=1)))
    true_std_all = float(np.mean(np.std(trues, axis=1)))
    diversity = pred_std_all / max(true_std_all, 1e-8)

    persistence_pred = np.broadcast_to(last_vals, trues.shape)
    persistence_mae = float(np.mean(np.abs(persistence_pred[mask] - trues[mask])))
    vs_persistence = mae / max(persistence_mae, 1e-8)

    step_mae = None
    if protocol == "paper":
        step_mae = []
        for step in range(trues.shape[1]):
            step_mask = mask[:, step, :]
            if step_mask.sum() == 0:
                step_mae.append(float("nan"))
                continue
            step_mae.append(float(np.mean(np.abs(preds[:, step, :][step_mask] - trues[:, step, :][step_mask]))))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "pred_std": pred_std_all,
        "true_std": true_std_all,
        "diversity": diversity,
        "persistence_mae": persistence_mae,
        "vs_persistence": vs_persistence,
        "step_mae": step_mae,
    }

def inverse_speed_channel(values, scaler):
    """Inverse-transform speed-channel arrays for both StandardScaler and MinMaxScaler.

    values is normally [B, T, N] or [B, 1, N]. The scaler was fit on [T, N, F],
    so only channel 0 statistics should be applied.
    """
    arr = np.asarray(values, dtype=np.float32)
    if hasattr(scaler, "mean") and getattr(scaler, "mean", None) is not None:
        mean = np.asarray(scaler.mean, dtype=np.float32)
        std = np.asarray(scaler.std, dtype=np.float32)
        if mean.ndim == 3:
            mean = mean[:, :, 0]
            std = std[:, :, 0]
        elif mean.ndim == 2:
            mean = mean[:, 0]
            std = std[:, 0]
        return arr * std + mean
    if hasattr(scaler, "min_val") and getattr(scaler, "min_val", None) is not None:
        min_val = np.asarray(scaler.min_val, dtype=np.float32)
        scale = np.asarray(scaler.scale, dtype=np.float32)
        if min_val.ndim == 3:
            min_val = min_val[:, :, 0]
            scale = scale[:, :, 0]
        elif min_val.ndim == 2:
            min_val = min_val[:, 0]
            scale = scale[:, 0]
        return arr * scale + min_val
    if hasattr(scaler, "inverse_transform"):
        return scaler.inverse_transform(arr)
    return arr

def build_status_record(
    dataset,
    pred_len,
    tag,
    target,
    result,
    protocol,
    frozen_tag="",
    active_tags=None,
    is_blocked=False,
    ckpt_mtime=None,
    eval_mtime=None,
    eval_split="test",
):
    active_tags = active_tags or []
    base = {
        "dataset": dataset,
        "pred_len": int(pred_len),
        "tag": tag or "",
        "protocol": protocol,
        "split": eval_split,
        "target": float(target),
        "mae": np.nan,
        "rmse": np.nan,
        "mape": np.nan,
        "gap_pct": np.nan,
        "diversity": np.nan,
        "persistence_mae": np.nan,
        "vs_persistence": np.nan,
        "n_samples": 0,
        "status": "NO_CKPT" if not tag else "ERROR",
        "error": "",
        "frozen_tag": frozen_tag or "",
        "active_tags": ";".join(sorted(set(active_tags))),
        "blocked": 1 if is_blocked else 0,
        "ckpt_mtime": ckpt_mtime if ckpt_mtime is not None else "",
        "eval_mtime": eval_mtime if eval_mtime is not None else int(time.time()),
    }
    if result is None:
        base["error"] = "no_result"
        return base
    if "error" in result:
        base["error"] = str(result["error"])
        return base

    status, gap = classify_status(result["mae"], target, result["diversity"], result["vs_persistence"])
    base.update({
        "mae": float(result["mae"]),
        "rmse": float(result["rmse"]),
        "mape": float(result["mape"]),
        "gap_pct": float(gap),
        "diversity": float(result["diversity"]),
        "persistence_mae": float(result["persistence_mae"]),
        "vs_persistence": float(result["vs_persistence"]),
        "n_samples": int(result.get("n_samples", 0)),
        "status": status,
        "error": "",
    })
    return base

def load_frozen_map(path="checkpoints/FROZEN.txt"):
    frozen = {}
    if not os.path.exists(path):
        return frozen
    with open(path, errors="ignore") as fh:
        for line in fh:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    frozen[(parts[0], int(parts[1]))] = parts[2]
                except ValueError:
                    continue
    return frozen

def load_blocked_triplets(path="checkpoints/BLOCKED.txt"):
    blocked = set()
    if not os.path.exists(path):
        return blocked
    with open(path, errors="ignore") as fh:
        for line in fh:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    blocked.add((parts[0], int(parts[1]), parts[2]))
                except ValueError:
                    continue
    return blocked

def discover_active_tags():
    active = {}
    try:
        out = subprocess.run(["ps", "-ww", "-eo", "args="], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    except Exception:
        return active
    for line in out.splitlines():
        if "train_metrla_optimized.py" not in line:
            continue
        ds = re.search(r"--dataset\s+(\S+)", line)
        pred = re.search(r"--pred_len\s+(\S+)", line)
        tag = re.search(r"--tag\s+(\S+)", line)
        if not (ds and pred and tag):
            continue
        try:
            key = (ds.group(1), int(pred.group(1)))
        except ValueError:
            continue
        active.setdefault(key, set()).add(tag.group(1))
    return active

def find_best_checkpoint(ds, pred_len):
    """Find checkpoint with lowest val MAE from progress logs."""
    ds_lower = ds.lower().replace("-", "")
    patterns = [
        f"logs/{ds_lower}_{pred_len}_*_progress.log",
        os.path.join(PACKAGE_ROOT, "results", "stman_logs", f"{ds_lower}_{pred_len}_*_progress.log"),
    ]
    best_mae = float("inf")
    best_tag = None

    for pattern in patterns:
        for logf in glob.glob(pattern):
            bn = os.path.basename(logf)
            # Extract tag: e.g. pemsbay_12_v6r_progress.log -> v6r
            m = re.match(rf"{ds_lower}_{pred_len}_(.+)_progress\.log", bn)
            if not m:
                continue
            tag = m.group(1)
            # Read last line
            with open(logf) as f:
                lines = f.readlines()
            if not lines:
                continue
            last = lines[-1].strip()
            m2 = re.search(r"best_MAE=([0-9.]+)", last)
            if not m2:
                continue
            mae = float(m2.group(1))
            if mae < best_mae:
                best_mae = mae
                best_tag = tag
    return best_tag, best_mae

def resolve_eval_tag(ds, pred_len, frozen_map, find_best=find_best_checkpoint):
    """Resolve the canonical tag to evaluate.

    Frozen cells are explicit source-of-record decisions and must not be
    displaced by a later exploratory tag whose progress log has a lower value.
    """
    frozen_tag = frozen_map.get((ds, pred_len), "")
    if frozen_tag:
        return frozen_tag, "frozen", None
    best_tag, best_mae = find_best(ds, pred_len)
    return best_tag, "best_log", best_mae

def detect_model_config(ckpt):
    """Detect model config from checkpoint."""
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    cfg = ckpt.get("config", {})

    # Detect d_model from state dict
    d_model = cfg.get("d_model", 64)
    detected_in_dim = cfg.get("in_dim", 2)
    for k, v in state.items():
        if "in_proj.weight" in k and v.dim() == 2:
            d_model = v.shape[0]
            detected_in_dim = v.shape[1]
            break

    # Detect use_direct_pred
    has_chunk = any("chunk_emb" in k for k in state)
    has_linear_direct = any("out_proj_direct" in k for k in state)
    has_temporal_agg = any("temporal_agg" in k for k in state)
    has_temporal_chunk_proj = any("temporal_chunk_proj" in k for k in state)
    has_attn_direct = any("chunk_query" in k or "temporal_key" in k for k in state)
    use_direct = cfg.get(
        "use_direct_pred",
        has_linear_direct or has_attn_direct or (has_chunk and (has_temporal_agg or has_temporal_chunk_proj)),
    )
    direct_head_mode = cfg.get(
        "direct_head_mode",
        "linear" if has_linear_direct else (
            "attn" if has_attn_direct else ("chunkwise" if has_temporal_chunk_proj else "legacy")
        ),
    )
    direct_step_refine = cfg.get(
        "direct_step_refine",
        any("chunk_step_emb.weight" in k or "step_refine." in k for k in state),
    )
    has_gated_residual = any("residual_gate" in k for k in state)
    gated_residual = cfg.get("gated_residual", has_gated_residual)
    scale_target_only = cfg.get("scale_target_only", False)
    calendar_features = cfg.get("calendar_features", "tod")
    calendar_count = 2 if calendar_features == "tod+dow" else (1 if calendar_features == "tod" else 0)
    has_future_time = any("future_time_proj" in k or "future_step_time_proj" in k for k in state)
    legacy_decoder_calendar = (
        not use_direct
        and "use_future_time_features" not in cfg
        and calendar_count > 0
        and detected_in_dim > calendar_count
    )
    use_future_time_features = cfg.get("use_future_time_features", has_future_time or legacy_decoder_calendar)
    decoder_future_tod = cfg.get(
        "decoder_future_tod",
        True if (use_direct or legacy_decoder_calendar) else False,
    )
    future_time_dim = None
    for k, v in state.items():
        if ("future_time_proj.weight" in k or "future_step_time_proj.weight" in k) and getattr(v, "ndim", None) == 2:
            future_time_dim = int(v.shape[1])
            break

    # Detect spatial_mode / k_hop from checkpoint state when old configs omitted them.
    spatial_mode = cfg.get("spatial_mode", "attention")
    k_hop = cfg.get("k_hop", 8)
    hop_sizes = sorted({
        int(v.numel())
        for k, v in state.items()
        if k.endswith("hop_weights") and getattr(v, "ndim", None) == 1
    })
    if hop_sizes:
        spatial_mode = cfg.get("spatial_mode", "gcn")
        if "k_hop" not in cfg and len(hop_sizes) == 1 and hop_sizes[0] >= 1:
            k_hop = hop_sizes[0] - 1

    # Detect encoder layers
    enc_layers = cfg.get("enc_layers", 3)
    max_enc = 0
    for k in state:
        m = re.match(r"st_encoder\.(\d+)\.", k)
        if m:
            max_enc = max(max_enc, int(m.group(1)) + 1)
    if max_enc > 0:
        enc_layers = max_enc

    n_heads = cfg.get("n_heads", 4)
    seq_len = cfg.get("seq_len", 12)
    in_dim = detected_in_dim
    scaler_type = cfg.get("scaler", "standard")
    ff_multiplier = cfg.get("ff_mult", cfg.get("ff_multiplier", 4))
    dropout = cfg.get("dropout", 0.0)
    if "time_feature_start" in cfg:
        time_feature_start = int(cfg["time_feature_start"])
    elif future_time_dim is not None:
        time_feature_start = max(0, in_dim - future_time_dim)
    elif use_future_time_features:
        time_feature_start = max(0, in_dim - calendar_count) if calendar_count else 1
    else:
        time_feature_start = 1

    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "enc_layers": enc_layers,
        "use_direct_pred": use_direct,
        "direct_head_mode": direct_head_mode,
        "direct_step_refine": direct_step_refine,
        "decoder_future_tod": decoder_future_tod,
        "gated_residual": gated_residual,
        "scale_target_only": scale_target_only,
        "calendar_features": calendar_features,
        "use_future_time_features": use_future_time_features,
        "time_feature_start": time_feature_start,
        "spatial_mode": spatial_mode,
        "k_hop": k_hop,
        "seq_len": seq_len,
        "in_dim": in_dim,
        "scaler": scaler_type,
        "ff_multiplier": ff_multiplier,
        "dropout": dropout,
    }

def evaluate_checkpoint(ds, pred_len, tag, raw, device="cpu", max_batches=30, protocol="current", split="test"):
    ckpt_path = f"checkpoints/{ds}_{pred_len}_{tag}_best.pt"
    if not os.path.exists(ckpt_path):
        return None

    data_dir = f"Datasets/{ds}"
    A = load_adjacency_csv(os.path.join(data_dir, "adj_matrix.csv"),
                           normalize="sym", add_self_loops=True).to(device)
    N = A.shape[0]
    SF = torch.tensor(np.load(f"spatial_fractal_vectors_{ds}.npy"), dtype=torch.float32).to(device)
    TF = torch.tensor(np.load(f"train_timefractals_{ds}.npy"), dtype=torch.float32).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = detect_model_config(ckpt)

    # Check in_dim matches data
    actual_in_dim = raw.shape[-1]
    if mcfg["in_dim"] != actual_in_dim and (
        mcfg["calendar_features"] == "tod+dow" or mcfg["in_dim"] == actual_in_dim + 1
    ):
        raw = load_raw_data(ds, add_dow=True)
        actual_in_dim = raw.shape[-1]
    if mcfg["in_dim"] != actual_in_dim:
        return {"error": f"in_dim mismatch: model={mcfg['in_dim']}, data={actual_in_dim}"}

    X_train, X_val, X_test = split_raw_by_protocol(raw, protocol)

    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        X_train, X_val, X_test,
        seq_len=mcfg["seq_len"], pred_len=pred_len,
        batch_size=4, scaler_type=mcfg["scaler"],
        scale_target_only=mcfg["scale_target_only"], target_dim=1)
    eval_loader = select_eval_loader(split, train_loader, val_loader, test_loader)

    model = STFractalTransformer(
        n_nodes=N, in_dim=actual_in_dim, out_dim=1,
        seq_len=mcfg["seq_len"], pred_len=pred_len,
        sf_dim=SF.shape[1], tf_dim=TF.shape[1],
        d_model=mcfg["d_model"], n_heads=mcfg["n_heads"],
        num_encoder_layers=mcfg["enc_layers"], num_decoder_layers=2,
        dropout=0.0, ff_multiplier=mcfg["ff_multiplier"],
        use_direct_pred=mcfg["use_direct_pred"],
        direct_head_mode=mcfg["direct_head_mode"],
        direct_step_refine=mcfg["direct_step_refine"],
        decoder_future_tod=mcfg["decoder_future_tod"],
        gated_residual=mcfg["gated_residual"],
        use_future_time_features=mcfg["use_future_time_features"],
        time_feature_start=mcfg["time_feature_start"],
        spatial_mode=mcfg["spatial_mode"], k_hop=mcfg["k_hop"])

    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    new_state = {k.replace("module.", "").replace("inner.", ""): v for k, v in state.items()}
    try:
        model.load_state_dict(new_state, strict=True)
    except RuntimeError:
        try:
            model.load_state_dict(new_state, strict=False)
        except Exception as e:
            return {"error": f"load failed: {e}"}

    model.to(device).eval()

    all_preds, all_trues, all_lastvals = [], [], []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(eval_loader):
            if max_batches > 0 and i >= max_batches:
                break
            yh, _ = model(xb.to(device), A, SF, TF)
            pn = yh[:, :, :, 0].cpu().numpy()
            tn = yb[:, :, :, 0].cpu().numpy()
            pi = inverse_speed_channel(pn, scaler)
            ti = inverse_speed_channel(tn, scaler)
            # persistence: last input step's speed channel
            lv_np = xb[:, -1:, :, 0].cpu().numpy()
            lv_inv = inverse_speed_channel(lv_np, scaler)
            all_preds.append(pi)
            all_trues.append(ti)
            all_lastvals.append(lv_inv)

    # Free GPU memory
    del model, A, SF, TF, ckpt, state, new_state
    if device != "cpu":
        torch.cuda.empty_cache()

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    last_vals = np.concatenate(all_lastvals, axis=0)  # [B, 1, N]

    metrics = compute_eval_metrics(preds, trues, last_vals, protocol=protocol)
    if "error" in metrics:
        return metrics

    return {
        "mae": metrics["mae"], "rmse": metrics["rmse"], "mape": metrics["mape"],
        "pred_std": metrics["pred_std"], "true_std": metrics["true_std"],
        "diversity": metrics["diversity"],
        "persistence_mae": metrics["persistence_mae"],
        "vs_persistence": metrics["vs_persistence"],
        "step_mae": metrics["step_mae"],
        "n_samples": preds.shape[0],
        "config": mcfg,
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1, help="GPU id for inference")
    ap.add_argument("--max_batches", type=int, default=None, help="Max test batches per checkpoint")
    ap.add_argument("--protocol", type=str, default="current", choices=["current", "paper"],
                    help="Evaluation protocol: current=internal 6:2:2, paper=7:1:2 + masked full-test")
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"],
                    help="Evaluation split")
    ap.add_argument("--dataset", type=str, default=None, help="Evaluate a single dataset")
    ap.add_argument("--pred_len", type=int, default=None, help="Evaluate a single horizon")
    ap.add_argument("--tag", type=str, default=None, help="Evaluate a single tag")
    ap.add_argument("--write_status", action="store_true",
                    help="Write machine-readable evaluation state to results/status.csv")
    ap.add_argument("--status_path", type=str, default="results/status.csv",
                    help="Path used with --write_status")
    cargs = ap.parse_args()
    if cargs.gpu < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cargs.gpu}")
    max_batches = resolve_max_batches(cargs.protocol, cargs.max_batches)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    frozen_map = load_frozen_map()
    blocked_triplets = load_blocked_triplets()
    active_tags = discover_active_tags()
    eval_mtime = int(time.time())

    if cargs.dataset and cargs.pred_len and cargs.tag:
        ds = cargs.dataset
        pred = cargs.pred_len
        tag = cargs.tag
        target = TARGETS.get((ds, pred), 999)
        try:
            raw = load_raw_data(ds)
            res = evaluate_checkpoint(ds, pred, tag, raw, device, max_batches, cargs.protocol, cargs.split)
        except Exception as e:
            print(f"SINGLE_RESULT dataset={ds} pred={pred} tag={tag} status=ERROR error={str(e).replace(' ', '_')}")
            return

        ckpt_file = f"checkpoints/{ds}_{pred}_{tag}_best.pt"
        ckpt_mtime = int(os.path.getmtime(ckpt_file)) if os.path.exists(ckpt_file) else ""
        status_record = build_status_record(
            dataset=ds,
            pred_len=pred,
            tag=tag,
            target=target,
            result=res,
            protocol=cargs.protocol,
            frozen_tag=frozen_map.get((ds, pred), ""),
            active_tags=active_tags.get((ds, pred), []),
            is_blocked=(ds, pred, tag) in blocked_triplets,
            ckpt_mtime=ckpt_mtime,
            eval_mtime=eval_mtime,
            eval_split=cargs.split,
        )
        if cargs.write_status:
            os.makedirs(os.path.dirname(cargs.status_path), exist_ok=True)
            pd.DataFrame([status_record]).to_csv(cargs.status_path, index=False)

        if res is None:
            print(f"SINGLE_RESULT dataset={ds} pred={pred} tag={tag} status=ERROR error=no_result")
            return
        if "error" in res:
            err = str(res["error"]).replace(" ", "_")
            print(f"SINGLE_RESULT dataset={ds} pred={pred} tag={tag} status=ERROR error={err}")
            return

        status, gap = classify_status(res["mae"], target, res["diversity"], res["vs_persistence"])
        step_mae = ""
        if res.get("step_mae") is not None:
            step_mae = " step_mae=" + ",".join(
                "nan" if np.isnan(v) else f"{v:.4f}" for v in res["step_mae"]
            )
        print(
            "SINGLE_RESULT "
            f"dataset={ds} pred={pred} tag={tag} "
            f"protocol={cargs.protocol} split={cargs.split} "
            f"mae={res['mae']:.4f} rmse={res['rmse']:.4f} mape={res['mape']:.2f} "
            f"target={target:.2f} gap_pct={gap:.2f} "
            f"div={res['diversity']:.4f} pers_mae={res['persistence_mae']:.4f} "
            f"vsp={res['vs_persistence']:.4f} samples={res['n_samples']} status={status}{step_mae}"
        )
        return

    # Skip currently running experiments
    running = {
        ("METR-LA", 12, "v12"), ("METR-LA", 48, "v12"), ("METR-LA", 2016, "v12dp"),
        ("METR-LA", 864, "v12bdp"),
        ("PEMS-BAY", 12, "v12"), ("PEMS-BAY", 48, "v12"), ("PEMS-BAY", 96, "v12"),
        ("PEMS-BAY", 288, "v12dp"), ("PEMS-BAY", 864, "v12dp"),
        ("PEMS07", 12, "v12"), ("PEMS07", 288, "v11dp"),
        ("PEMS08", 864, "v12dp"),
    }

    datasets = ["METR-LA", "PEMS-BAY", "PEMS03", "PEMS07", "PEMS08"]
    horizons = [12, 48, 96, 288, 864, 2016]

    # Pre-load raw data per dataset
    raw_data = {}
    for ds in datasets:
        print(f"Loading {ds} data...", flush=True)
        raw_data[ds] = load_raw_data(ds)

    print(f"\n{'='*130}")
    print(f"{'Dataset':<10} {'Pred':>5} {'Tag':<12} {'MAE':>8} {'MAPE%':>8} "
          f"{'Target':>8} {'Gap%':>7} {'Div':>6} {'PersMAE':>8} {'vs.Pers':>8} {'Status':<14}")
    print(f"{'='*130}")

    results = []
    status_records = []
    for ds in datasets:
        for pred in horizons:
            target = TARGETS.get((ds, pred), 999)
            tag, tag_source, log_mae = resolve_eval_tag(ds, pred, frozen_map)
            frozen_tag = frozen_map.get((ds, pred), "")
            active_for_cell = active_tags.get((ds, pred), [])
            if tag is None:
                print(f"{ds:<10} {pred:>5} {'N/A':<12} {'---':>8} {'---':>8} {'---':>8} "
                      f"{target:>8.2f} {'---':>7} {'---':>10} {'no ckpt':<12}")
                status_records.append(build_status_record(
                    dataset=ds, pred_len=pred, tag=None, target=target, result=None,
                    protocol=cargs.protocol, frozen_tag=frozen_tag, active_tags=active_for_cell,
                    is_blocked=False, ckpt_mtime="", eval_mtime=eval_mtime,
                    eval_split=cargs.split,
                ))
                continue

            # Skip running experiments, use next best
            if (ds, pred, tag) in running:
                # Find next best completed checkpoint
                ds_lower = ds.lower().replace("-", "")
                pattern = f"logs/{ds_lower}_{pred}_*_progress.log"
                candidates = []
                for logf in glob.glob(pattern):
                    bn = os.path.basename(logf)
                    m = re.match(rf"{ds_lower}_{pred}_(.+)_progress\.log", bn)
                    if not m: continue
                    t = m.group(1)
                    if (ds, pred, t) in running: continue
                    with open(logf) as f:
                        lines = f.readlines()
                    if not lines: continue
                    m2 = re.search(r"best_MAE=([0-9.]+)", lines[-1])
                    if m2: candidates.append((float(m2.group(1)), t))
                if candidates:
                    candidates.sort()
                    tag = candidates[0][1]
                    log_mae = candidates[0][0]
                else:
                    print(f"{ds:<10} {pred:>5} {'running':<12} {'---':>8} {'---':>8} {'---':>8} "
                          f"{target:>8.2f} {'---':>7} {'---':>10} {'all running':<12}")
                    status_records.append(build_status_record(
                        dataset=ds, pred_len=pred, tag=tag, target=target,
                        result={"error": "all_running"},
                        protocol=cargs.protocol, frozen_tag=frozen_tag, active_tags=active_for_cell,
                        is_blocked=(ds, pred, tag) in blocked_triplets,
                        ckpt_mtime="", eval_mtime=eval_mtime,
                        eval_split=cargs.split,
                    ))
                    continue

            ckpt_file = f"checkpoints/{ds}_{pred}_{tag}_best.pt"
            ckpt_mtime = int(os.path.getmtime(ckpt_file)) if os.path.exists(ckpt_file) else ""
            if not os.path.exists(ckpt_file):
                print(f"{ds:<10} {pred:>5} {tag:<12} {'---':>8} {'---':>8} {'---':>8} "
                      f"{target:>8.2f} {'---':>7} {'---':>10} {'no file':<12}")
                status_records.append(build_status_record(
                    dataset=ds, pred_len=pred, tag=tag, target=target,
                    result={"error": "no_checkpoint_file"},
                    protocol=cargs.protocol, frozen_tag=frozen_tag, active_tags=active_for_cell,
                    is_blocked=(ds, pred, tag) in blocked_triplets,
                    ckpt_mtime="", eval_mtime=eval_mtime,
                    eval_split=cargs.split,
                ))
                continue

            print(f"{ds:<10} {pred:>5} {tag:<12} evaluating...", end="\r", flush=True)
            try:
                res = evaluate_checkpoint(ds, pred, tag, raw_data[ds], device, max_batches, cargs.protocol, cargs.split)
            except Exception as e:
                print(f"{ds:<10} {pred:>5} {tag:<12} {'ERROR':>8} {str(e)[:50]}")
                status_records.append(build_status_record(
                    dataset=ds, pred_len=pred, tag=tag, target=target,
                    result={"error": str(e)},
                    protocol=cargs.protocol, frozen_tag=frozen_tag, active_tags=active_for_cell,
                    is_blocked=(ds, pred, tag) in blocked_triplets,
                    ckpt_mtime=ckpt_mtime, eval_mtime=eval_mtime,
                    eval_split=cargs.split,
                ))
                continue

            if res is None or "error" in res:
                err = res.get("error", "unknown") if res else "None"
                print(f"{ds:<10} {pred:>5} {tag:<12} {'ERROR':>8} {err[:50]}")
                status_records.append(build_status_record(
                    dataset=ds, pred_len=pred, tag=tag, target=target,
                    result=res or {"error": "None"},
                    protocol=cargs.protocol, frozen_tag=frozen_tag, active_tags=active_for_cell,
                    is_blocked=(ds, pred, tag) in blocked_triplets,
                    ckpt_mtime=ckpt_mtime, eval_mtime=eval_mtime,
                    eval_split=cargs.split,
                ))
                continue

            mae = res["mae"]
            status, gap = classify_status(mae, target, res["diversity"], res["vs_persistence"])
            div = res["diversity"]
            pers_mae = res["persistence_mae"]
            vs_p = res["vs_persistence"]

            print(f"{ds:<10} {pred:>5} {tag:<12} {mae:>8.2f} {res['mape']:>7.1f}% "
                      f"{target:>8.2f} {gap:>+6.1f}% {div:>6.2f} {pers_mae:>8.2f} {vs_p:>7.2f}x {status:<14}")

            results.append({
                "dataset": ds, "pred_len": pred, "tag": tag,
                "mae": mae, "rmse": res["rmse"], "mape": res["mape"],
                "target": target, "gap_pct": gap, "diversity": div,
                "persistence_mae": pers_mae, "vs_persistence": vs_p,
                "status": status
            })
            status_records.append(build_status_record(
                dataset=ds,
                pred_len=pred,
                tag=tag,
                target=target,
                result=res,
                protocol=cargs.protocol,
                frozen_tag=frozen_tag,
                active_tags=active_for_cell,
                is_blocked=(ds, pred, tag) in blocked_triplets,
                ckpt_mtime=ckpt_mtime,
                eval_mtime=eval_mtime,
                eval_split=cargs.split,
            ))

    # Summary
    print(f"\n{'='*130}")
    true_r = sum(1 for r in results if r["status"] == "TRUE_REACHED")
    border = sum(1 for r in results if r["status"] == "BORDERLINE")
    collapsed = sum(1 for r in results if r["status"] == "COLLAPSED")
    close = sum(1 for r in results if r["status"] == "CLOSE")
    mod = sum(1 for r in results if r["status"] == "MODERATE")
    far = sum(1 for r in results if r["status"] == "FAR")
    total = len(results)
    print(f"Summary: {true_r} TRUE_REACHED, {border} BORDERLINE, {collapsed} COLLAPSED, "
          f"{close} CLOSE, {mod} MODERATE, {far} FAR / {total} evaluated")
    if cargs.write_status:
        os.makedirs(os.path.dirname(cargs.status_path), exist_ok=True)
        pd.DataFrame(status_records).to_csv(cargs.status_path, index=False)
        print(f"Status written to {cargs.status_path} ({len(status_records)} rows)")

if __name__ == "__main__":
    main()
