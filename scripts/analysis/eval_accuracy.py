#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_accuracy.py — 批量评估已完成实验的测试集精度
指标: MAE / RMSE / MAPE / ACC@10% / ACC@15%

ACC@k% = 非零样本中 |pred-true| / max(|true|, 1.0) <= k/100 的比例
(分母与 train.py 中 MAPE 计算保持一致，避免极小值放大误差)

用法:
  python eval_accuracy.py                     # 跑全部内置实验
  python eval_accuracy.py --ckpt PEMS08_12_v9_best.pt --ds PEMS08 --scaler standard
  python eval_accuracy.py --tag v6            # 只跑包含 "v6" 的 checkpoint
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_adjacency_csv, create_data_loaders
from model import STFractalTransformer

WORKDIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = "/mnt/nfs/yxwang/env/fract_env/bin/python"

# ── 每个数据集的默认配置 ──────────────────────────────────────────────────────
DATASET_CONF = {
    "METR-LA":  {"scaler": "minmax",   "steps_per_day": 288, "split_rate": 0.6},
    "PEMS-BAY": {"scaler": "minmax",   "steps_per_day": 288, "split_rate": 0.6},
    "PEMS03":   {"scaler": "standard", "steps_per_day": 288, "split_rate": 0.6},
    "PEMS07":   {"scaler": "standard", "steps_per_day": 288, "split_rate": 0.6},
    "PEMS08":   {"scaler": "standard", "steps_per_day": 288, "split_rate": 0.6},
}

# ── 需要评估的实验列表 (checkpoint文件名, 数据集, scaler覆盖或None)
# 格式: (ckpt_basename, dataset, scaler_override_or_None)
EVAL_LIST = [
    # METR-LA
    ("METR-LA_12_v6_best.pt",    "METR-LA",  None),
    ("METR-LA_48_v6_best.pt",    "METR-LA",  None),
    ("METR-LA_96_v6_best.pt",    "METR-LA",  None),
    ("METR-LA_288_v6_best.pt",   "METR-LA",  None),
    ("METR-LA_864_v6dp_best.pt", "METR-LA",  None),
    ("METR-LA_2016_v6_best.pt",  "METR-LA",  None),
    # PEMS-BAY
    ("PEMS-BAY_12_v6_best.pt",   "PEMS-BAY", None),
    ("PEMS-BAY_48_v6_best.pt",   "PEMS-BAY", None),
    ("PEMS-BAY_96_v6_best.pt",   "PEMS-BAY", None),
    ("PEMS-BAY_96_v7_best.pt",   "PEMS-BAY", None),
    ("PEMS-BAY_864_v6_best.pt",  "PEMS-BAY", None),
    ("PEMS-BAY_2016_v6_best.pt", "PEMS-BAY", None),
    # PEMS03
    ("PEMS03_12_v6_best.pt",     "PEMS03",   None),
    ("PEMS03_12_v6c_best.pt",    "PEMS03",   None),
    ("PEMS03_48_v6_best.pt",     "PEMS03",   None),
    ("PEMS03_48_v6c_best.pt",    "PEMS03",   None),
    ("PEMS03_48_v6d_best.pt",    "PEMS03",   None),
    ("PEMS03_864_v6_best.pt",    "PEMS03",   None),
    ("PEMS03_2016_v6_best.pt",   "PEMS03",   None),
    # PEMS07
    ("PEMS07_12_v6_best.pt",     "PEMS07",   None),
    ("PEMS07_12_v6c_best.pt",    "PEMS07",   None),
    ("PEMS07_48_v6c_best.pt",    "PEMS07",   None),
    ("PEMS07_48_v7_best.pt",     "PEMS07",   None),
    ("PEMS07_48_v7c_best.pt",    "PEMS07",   None),
    ("PEMS07_96_v6c_best.pt",    "PEMS07",   None),
    ("PEMS07_288_v6c_best.pt",   "PEMS07",   None),
    ("PEMS07_2016_v6_best.pt",   "PEMS07",   None),
    # PEMS08
    ("PEMS08_12_v6_best.pt",     "PEMS08",   None),
    ("PEMS08_12_v6c_best.pt",    "PEMS08",   None),
    ("PEMS08_12_v7_best.pt",     "PEMS08",   None),
    ("PEMS08_12_v8_best.pt",     "PEMS08",   None),
    ("PEMS08_12_v8s_best.pt",    "PEMS08",   None),
    ("PEMS08_12_v9_best.pt",     "PEMS08",   None),
    ("PEMS08_48_v6_best.pt",     "PEMS08",   None),
    ("PEMS08_96_v6c_best.pt",    "PEMS08",   None),
    ("PEMS08_96_v6cs_best.pt",   "PEMS08",   None),
    ("PEMS08_96_v9_best.pt",     "PEMS08",   None),
    ("PEMS08_864_v6_best.pt",    "PEMS08",   None),
]

# ── 论文目标 MAE ──────────────────────────────────────────────────────────────
TARGET_MAE = {
    ("METR-LA",  12): 3.28,  ("METR-LA",  48): 5.19,  ("METR-LA",  96): 7.80,
    ("METR-LA", 288): 9.44,  ("METR-LA", 864): 10.50, ("METR-LA", 2016): 11.13,
    ("PEMS-BAY",  12): 1.86, ("PEMS-BAY",  48): 2.59, ("PEMS-BAY",  96): 2.78,
    ("PEMS-BAY", 288): 2.98, ("PEMS-BAY", 864): 3.17, ("PEMS-BAY", 2016): 3.39,
    ("PEMS03",  12): 14.42,  ("PEMS03",  48): 15.36,  ("PEMS03",  96): 17.04,
    ("PEMS03", 288): 18.36,  ("PEMS03", 864): 20.09,  ("PEMS03", 2016): 22.92,
    ("PEMS07",  12): 22.32,  ("PEMS07",  48): 24.07,  ("PEMS07",  96): 25.90,
    ("PEMS07", 288): 27.63,  ("PEMS07", 864): 29.73,  ("PEMS07", 2016): 31.99,
    ("PEMS08",  12): 13.75,  ("PEMS08",  48): 17.31,  ("PEMS08",  96): 21.76,
    ("PEMS08", 288): 22.37,  ("PEMS08", 864): 24.46,  ("PEMS08", 2016): 26.23,
}


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_raw(ds: str) -> np.ndarray:
    data_dir = os.path.join(WORKDIR, "Datasets", ds)
    npz  = os.path.join(data_dir, f"{ds}.npz")
    h5   = os.path.join(data_dir, "metr-la.h5")   # METR-LA 专用
    csv  = os.path.join(data_dir, f"{ds}.csv")

    if os.path.exists(npz):
        raw = np.load(npz)["data"].astype(np.float32)
    elif os.path.exists(h5):
        raw = pd.read_hdf(h5).values.astype(np.float32)
    elif os.path.exists(csv):
        raw = pd.read_csv(csv, header=None).values.astype(np.float32)
    else:
        raise FileNotFoundError(f"No data file for {ds} in {data_dir}")

    if raw.ndim == 2:
        raw = raw[:, :, np.newaxis]   # [T, N, 1]

    # 时间特征 (time-of-day)
    T = raw.shape[0]
    spd = DATASET_CONF[ds]["steps_per_day"]
    tod = np.array([(i % spd) / spd for i in range(T)], dtype=np.float32)
    tod = np.broadcast_to(tod[:, None, None], (T, raw.shape[1], 1)).copy()
    return np.concatenate([raw, tod], axis=-1)    # [T, N, F+1]


# ── 指标计算 ──────────────────────────────────────────────────────────────────
def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE / RMSE / MAPE / ACC@10% / ACC@15%，排除零值"""
    mask  = np.abs(y_true) > 1e-4
    if mask.sum() == 0:
        return dict(mae=0., rmse=0., mape=0., acc10=0., acc15=0.)

    diff    = y_pred[mask] - y_true[mask]
    denom   = np.maximum(np.abs(y_true[mask]), 1.0)   # 与 train.py MAPE 一致
    rel_err = np.abs(diff) / denom

    return dict(
        mae   = float(np.mean(np.abs(diff))),
        rmse  = float(np.sqrt(np.mean(diff ** 2))),
        mape  = float(np.mean(rel_err)),
        acc10 = float(np.mean(rel_err <= 0.10)),
        acc15 = float(np.mean(rel_err <= 0.15)),
    )


def inverse_ch0(arr: np.ndarray, scaler) -> np.ndarray:
    """对 channel-0 (speed) 做反标准化；arr shape: [..., N]"""
    if scaler is None:
        return arr
    if hasattr(scaler, "mean"):            # StandardScaler
        mu  = scaler.mean[0, :, 0]        # [N]
        std = scaler.std[0, :, 0]
        return arr * std + mu
    elif hasattr(scaler, "min_val"):       # MinMaxScaler
        mn  = scaler.min_val[0, :, 0]
        sc  = scaler.scale[0, :, 0]
        return arr * sc + mn
    return arr


# ── 单 checkpoint 评估 ────────────────────────────────────────────────────────
def evaluate_one(ckpt_path: str, ds: str, scaler_type: str,
                 device: torch.device, batch_size: int = 64) -> dict:
    ck  = torch.load(ckpt_path, map_location="cpu")
    cfg = ck["config"]
    seq_len  = cfg["seq_len"]
    pred_len = cfg["pred_len"]
    in_dim   = cfg["in_dim"]
    epoch    = ck.get("epoch", "?")

    # 自动检测旧/新 direct head，兼容早期 out_proj_direct checkpoint
    state = ck["model"]
    has_linear_direct = any("out_proj_direct" in k for k in state.keys())
    has_chunk = any("chunk_emb" in k for k in state.keys())
    has_temporal_agg = any("temporal_agg" in k for k in state.keys())
    has_temporal_chunk_proj = any("temporal_chunk_proj" in k for k in state.keys())
    use_direct_pred = has_linear_direct or (has_chunk and (has_temporal_agg or has_temporal_chunk_proj))
    direct_head_mode = cfg.get(
        "direct_head_mode",
        "linear" if has_linear_direct else ("chunkwise" if has_temporal_chunk_proj else "legacy"),
    )

    # 加载数据
    raw = load_raw(ds)
    if raw.shape[-1] != in_dim:
        # 如果 in_dim 不匹配，截断或填充（兼容旧版 checkpoint）
        if in_dim < raw.shape[-1]:
            raw = raw[:, :, :in_dim]
        else:
            # 用零填充（不应该发生）
            pad = np.zeros((raw.shape[0], raw.shape[1], in_dim - raw.shape[-1]), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=-1)

    split = DATASET_CONF[ds]["split_rate"]
    T = raw.shape[0]
    n_train = int(T * split)
    n_val   = int(T * 0.2)
    X_train = raw[:n_train]
    X_val   = raw[n_train:n_train + n_val]
    X_test  = raw[n_train + n_val:]

    _, _, test_loader, scaler = create_data_loaders(
        X_train, X_val, X_test,
        seq_len=seq_len, pred_len=pred_len,
        batch_size=batch_size, scaler_type=scaler_type, verbose=False,
    )

    # 构建模型
    A_path = os.path.join(WORKDIR, "Datasets", ds, "adj_matrix.csv")
    A  = load_adjacency_csv(A_path, normalize="sym", add_self_loops=True).to(device)
    SF = torch.tensor(np.load(os.path.join(WORKDIR, f"spatial_fractal_vectors_{ds}.npy")),
                      dtype=torch.float32).to(device)
    TF = torch.tensor(np.load(os.path.join(WORKDIR, f"train_timefractals_{ds}.npy")),
                      dtype=torch.float32).to(device)

    model = STFractalTransformer(
        n_nodes          = cfg["n_nodes"],
        in_dim           = in_dim,
        out_dim          = 1,
        d_model          = cfg["d_model"],
        n_heads          = cfg["n_heads"],
        num_encoder_layers = cfg["enc_layers"],
        num_decoder_layers = cfg["dec_layers"],
        seq_len          = seq_len,
        pred_len         = pred_len,
        sf_dim           = cfg["sf_dim"],
        tf_dim           = cfg["tf_dim"],
        dropout          = cfg["dropout"],
        ff_multiplier    = cfg["ff_mult"],
        use_direct_pred  = use_direct_pred,
        direct_head_mode = direct_head_mode,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat, _ = model(x, A, SF, TF)          # [B, pred, N, 1]
            # 取 channel-0 → [B, pred, N]
            preds.append(y_hat[..., 0].cpu().numpy())
            trues.append(y[..., 0].cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)           # [B, pred, N]
    y_true = np.concatenate(trues, axis=0)

    # 反标准化
    # scaler.mean/std/min_val/scale shape: [1, N, F]
    # 我们对 channel-0 做反变换，但 y_pred/y_true 形状是 [B, pred, N]
    # 逐 N 维广播
    if hasattr(scaler, "mean"):
        mu  = scaler.mean[0, :, 0]   # [N]
        std = scaler.std[0, :, 0]
        y_pred_inv = y_pred * std + mu
        y_true_inv = y_true * std + mu
    elif hasattr(scaler, "min_val"):
        mn = scaler.min_val[0, :, 0]
        sc = scaler.scale[0, :, 0]
        y_pred_inv = y_pred * sc + mn
        y_true_inv = y_true * sc + mn
    else:
        y_pred_inv = y_pred
        y_true_inv = y_true

    m = calc_metrics(y_true_inv, y_pred_inv)
    m["epoch"] = epoch
    m["use_direct"] = use_direct_pred
    return m


# ── 主程序 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   type=str, default=None, help="单个 checkpoint 文件名（basename）")
    parser.add_argument("--ds",     type=str, default=None, help="数据集名（--ckpt 时必填）")
    parser.add_argument("--scaler", type=str, default=None, help="scaler 覆盖（standard/minmax）")
    parser.add_argument("--tag",    type=str, default=None, help="只评估名称包含该 tag 的 checkpoint")
    parser.add_argument("--gpu",    type=int, default=0,    help="使用的 GPU ID")
    parser.add_argument("--batch",  type=int, default=64)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(WORKDIR, "checkpoints")

    if args.ckpt:
        # 单个 checkpoint 模式
        ckpt_path = os.path.join(ckpt_dir, args.ckpt)
        if not os.path.exists(ckpt_path):
            print(f"ERROR: {ckpt_path} not found"); sys.exit(1)
        if not args.ds:
            print("ERROR: --ds required with --ckpt"); sys.exit(1)
        scaler_type = args.scaler or DATASET_CONF[args.ds]["scaler"]
        m = evaluate_one(ckpt_path, args.ds, scaler_type, device, args.batch)
        print(f"\n{args.ckpt} [{args.ds}] ep={m['epoch']}")
        print(f"  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  MAPE={m['mape']*100:.2f}%"
              f"  ACC@10%={m['acc10']*100:.1f}%  ACC@15%={m['acc15']*100:.1f}%")
        return

    # 批量模式
    eval_list = EVAL_LIST
    if args.tag:
        eval_list = [(c, d, s) for c, d, s in eval_list if args.tag in c]

    rows = []
    header = f"{'Checkpoint':<35} {'DS':<10} {'Ep':>4}  {'MAE':>7}  {'RMSE':>7}  {'MAPE':>6}  {'ACC@10':>7}  {'ACC@15':>7}  {'Target':>7}  {'vs Target':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for ckpt_name, ds, scaler_override in eval_list:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"{'[SKIP]':<35} {ckpt_name}")
            continue

        scaler_type = scaler_override or DATASET_CONF[ds]["scaler"]
        try:
            m = evaluate_one(ckpt_path, ds, scaler_type, device, args.batch)
        except Exception as e:
            print(f"{'[ERROR]':<35} {ckpt_name}: {e}")
            continue

        # 取 pred_len from checkpoint name
        ck = torch.load(ckpt_path, map_location="cpu")
        pred_len = ck["config"]["pred_len"]
        target = TARGET_MAE.get((ds, pred_len), float("nan"))
        vs_t = f"{m['mae'] - target:+.3f}" if not np.isnan(target) else "  N/A"

        row = (f"{ckpt_name:<35} {ds:<10} {m['epoch']:>4}  "
               f"{m['mae']:>7.4f}  {m['rmse']:>7.4f}  "
               f"{m['mape']*100:>5.2f}%  "
               f"{m['acc10']*100:>6.1f}%  "
               f"{m['acc15']*100:>6.1f}%  "
               f"{target:>7.2f}  {vs_t:>10}")
        print(row)
        rows.append(dict(
            checkpoint=ckpt_name, dataset=ds, pred_len=pred_len,
            epoch=m["epoch"], mae=round(m["mae"],4), rmse=round(m["rmse"],4),
            mape_pct=round(m["mape"]*100, 2),
            acc10_pct=round(m["acc10"]*100, 1), acc15_pct=round(m["acc15"]*100, 1),
            target_mae=target, delta_vs_target=round(m["mae"]-target, 4),
        ))

    print("=" * len(header))

    if rows:
        df = pd.DataFrame(rows)
        out_csv = os.path.join(WORKDIR, "eval_accuracy_results.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nResults saved to: {out_csv}")

        # 汇总统计
        print("\n── ACC@10% 汇总（按数据集分组）─────────────────────────────────")
        for ds_name in df["dataset"].unique():
            sub = df[df["dataset"] == ds_name]
            print(f"  {ds_name}: mean ACC@10%={sub['acc10_pct'].mean():.1f}%  "
                  f"mean ACC@15%={sub['acc15_pct'].mean():.1f}%  "
                  f"mean MAPE={sub['mape_pct'].mean():.2f}%")


if __name__ == "__main__":
    main()
