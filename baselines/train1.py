# -*- coding: utf-8 -*-
#########train.py
import os
import time
from typing import Optional, Dict
from types import SimpleNamespace
from contextlib import nullcontext  # AMP 需要

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 原有模型
from model.TGCN import TGCNForecaster
from model.STGCN import STGCNChebGraphConv, STGCNGraphConv
# LSTNN（即 MultiscaleMLP 的顶层类）
from model.LSTNN import MultiscaleMLP
# === 新增：STAEformer ===
from model.STAEformer import STAEformer

# === 新增：PDFormer ===
from model.PDFormer import PDFormer  # 如果你把 pdformer.py 放在 model/ 下


from metric import HuberRegLoss
from utils import build_gso_for_stgcn  # STGCN 的 GSO 构造

# ----------------------------
# 小工具
# ----------------------------
def _compute_shortest_path_dist(A: np.ndarray) -> np.ndarray:
    """基于权重的最短路 (Floyd-Warshall)。A>=0，0 代表无边或权重为0（视为不可达）。"""
    N = A.shape[0]
    INF = 1e12
    W = A.copy().astype(np.float64)
    D = np.where(W > 0, W, INF)
    np.fill_diagonal(D, 0.0)
    for k in range(N):
        # 向量化松弛
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D

def _compute_hop_dist(A: np.ndarray) -> np.ndarray:
    """无权图最短跳数距离（把有边视为权重=1，Floyd-Warshall 一次到位）"""
    G = (A > 0).astype(np.float32)             # 二值化
    N = G.shape[0]
    INF = 1e12
    D = np.where(G > 0, 1.0, INF)              # 有边 -> 1
    np.fill_diagonal(D, 0.0)
    for k in range(N):                          # O(N^3)，但只做一次
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    D[D >= INF/2] = N + 1                       # 不可达：设为一个大值
    return D.astype(np.float32)

def _laplacian_positional_eigvecs(adj: torch.Tensor, k: int) -> torch.Tensor:
    """
    稳健版：计算归一化拉普拉斯 L = I - D^{-1/2} A D^{-1/2} 的前 k 个（最小）特征向量
    - 强制清洗 A：去 NaN/Inf、裁负数、对称化
    - 保证 L 严格对称：0.5*(L+L^T)
    - 用 torch.linalg.eigh 在 CPU/float64 上分解，失败则加 jitter 再试
    - 最后兜底：返回 I 的前 k 列
    """
    assert k >= 1, "k (lape_dim) 必须 >= 1"
    with torch.no_grad():
        # ---- 取到 CPU 上做数值稳定的分解 ----
        A = adj.detach().cpu().to(dtype=torch.float64).numpy()
        N = A.shape[0]
        if k > N:
            k = N

        # ---- 清洗 & 对称化 ----
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = np.clip(A, a_min=0.0, a_max=None)
        A = 0.5 * (A + A.T)
        # 可选：去除极端大权重（避免病态条件数）
        if np.isfinite(A).any():
            p99 = np.percentile(A[A > 0], 99) if np.any(A > 0) else 0.0
            if p99 > 0:
                A = np.minimum(A, p99)

        # ---- 归一化拉普拉斯 ----
        d = A.sum(axis=1)
        d = np.where(np.isfinite(d), d, 0.0) + 1e-12
        dinvs = 1.0 / np.sqrt(d)
        A_sym = (dinvs[:, None] * A) * dinvs[None, :]
        I = np.eye(N, dtype=np.float64)
        L = I - A_sym
        L = 0.5 * (L + L.T)  # 保证严格对称

        # ---- 尝试特征分解 ----
        def eigh_with_retry(M: np.ndarray, jitter: float):
            MM = M.copy()
            if jitter > 0:
                MM[np.arange(N), np.arange(N)] += jitter
            # 用 CPU / float64 做分解更稳
            M_t = torch.from_numpy(MM)
            evals, evecs = torch.linalg.eigh(M_t)  # 全量分解，N≈几百没问题
            return evals.numpy(), evecs.numpy()

        try:
            evals, evecs = eigh_with_retry(L, jitter=0.0)
        except Exception:
            try:
                # 轻微抖动，缓解半正定/近奇异
                evals, evecs = eigh_with_retry(L, jitter=1e-8)
            except Exception:
                # 兜底：不用位置编码，返回 I 的前 k 列（等价于没有空间 PE）
                eye_k = np.eye(N, k, dtype=np.float64)
                U = torch.from_numpy(eye_k).to(dtype=torch.float32, device=adj.device)
                return U

        # ---- 取最小的 k 个特征向量 ----
        idx = np.argsort(evals)[:k]
        U = evecs[:, idx].astype(np.float64)   # (N,k)
        U = torch.from_numpy(U).to(dtype=torch.float32, device=adj.device)
        return U


def _ensure_parent_dir(path: str):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)

def _init_epoch_log(log_path: str, headers):
    _ensure_parent_dir(log_path)
    new_file = not os.path.exists(log_path)
    f = open(log_path, "a", buffering=1, encoding="utf-8")
    if new_file:
        f.write("# " + "\t".join(headers) + "\n")
    return f

def _count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def _print_model_profile(model: nn.Module, **cfg):
    total, trainable = _count_parameters(model)
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    if cfg:
        print("Config: " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
    approx_mb = trainable * 4 / 1024 / 1024
    print(f"Parameters: total={total:,} | trainable={trainable:,} (~{trainable/1e6:.2f} M)")
    print(f"Approx. model size: {approx_mb:.2f} MB (fp32)")
    print("=" * 60)

def inverse_transform_batch(arr: torch.Tensor, scaler=None) -> np.ndarray:
    arr_np = arr.detach().cpu().numpy()
    if scaler is None:
        return arr_np
    B, T, N, F = arr_np.shape
    flat = arr_np.reshape(B * T, N, F)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(B, T, N, F)

def calc_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, scaler=None, eps: float = 1e-6) -> Dict[str, float]:
    yt = inverse_transform_batch(y_true, scaler)
    yp = inverse_transform_batch(y_pred, scaler)
    diff = yp - yt
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    denom = np.maximum((np.abs(yt) + np.abs(yp)) / 2.0, eps)
    smape = float(np.mean(np.abs(yp - yt) / denom))
    return {"mae": mae, "rmse": rmse, "smape": smape}

# ----------------------------
# ckpt
# ----------------------------
def _pack_config(**kwargs):
    return dict(**kwargs)

def save_checkpoint(path: str,
                    model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                    epoch: int,
                    monitor_value: float,
                    config: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "epoch": int(epoch),
        "val_loss": float(monitor_value),
        "config": config,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        try:
            ckpt["scheduler"] = scheduler.state_dict()
        except Exception:
            pass
    torch.save(ckpt, path)

def load_checkpoint(path: str,
                    model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                    map_location: Optional[torch.device] = None):
    ckpt = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val = ckpt.get("val_loss", float("inf"))
    cfg = ckpt.get("config", {})
    return start_epoch, best_val, cfg

# ----------------------------
# 训练与评估（支持 tgcn / stgcn / lstnn / staeformer / pdformer）
# ----------------------------
def _forward_bridge(
    model: nn.Module,
    x: torch.Tensor,           # [B, Tin, N, F]
    y: torch.Tensor,           # [B, Tout, N, F]
    model_kind: str,
    teacher_forcing_ratio: float,
) -> torch.Tensor:
    """把多种模型的前向统一成 [B, Tout, N, F]"""
    if model_kind == "tgcn":
        return model(x, y_teacher=y, teacher_forcing_ratio=teacher_forcing_ratio)

    if model_kind == "stgcn":
        x_st = x.permute(0, 3, 1, 2).contiguous()
        y_hat = model(x_st)                     # [B, EC, 1, N]
        B, EC, _, N = y_hat.shape
        H = y.size(1)
        F = y.size(3)
        if EC == H * F:
            y_hat = y_hat.squeeze(2).contiguous()        # [B, EC, N]
            y_hat = y_hat.reshape(B, H, F, N).permute(0, 1, 3, 2).contiguous()  # [B,H,N,F]
            return y_hat
        elif EC == H and F == 1:
            return y_hat.squeeze(2).unsqueeze(-1).contiguous()  # [B,H,N,1]
        else:
            raise RuntimeError(f"[STGCN] end_channel={EC} != H*F={H*F}，请把 end_channel 设为 pred_len*out_dim。")

    if model_kind == "lstnn":
        y_hat = model(x)                     # [B, H, N]
        y_hat = y_hat.unsqueeze(-1).contiguous()  # -> [B, H, N, 1]
        return y_hat

    if model_kind == "staeformer":
        y_hat = model(x)                     # -> [B, H, N, out_dim]
        return y_hat

    if model_kind == "pdformer":
        batch = {"X": x, "y": y}
        return model(batch, model._lap_mx)

    raise ValueError(f"Unknown model_kind: {model_kind}")

def train_one_epoch(
    model: nn.Module,
    criterion: HuberRegLoss,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    scaler_obj=None,
    grad_clip: Optional[float] = 1.0,
    teacher_forcing_ratio: float = 0.0,
    model_kind: str = "tgcn",
) -> Dict[str, float]:
    model.train()
    meter = {"loss": 0.0, "mae": 0.0, "rmse": 0.0, "smape": 0.0}
    seen = 0

    amp_enabled = (device.type == "cuda")
    autocast = torch.cuda.amp.autocast if amp_enabled else nullcontext
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    for x, y in loader:
        x = x.to(device)  # [B,Tin,N,F]
        y = y.to(device)  # [B,Tout,N,F]
        optimizer.zero_grad(set_to_none=True)

        with autocast(dtype=torch.float16):
            y_hat = _forward_bridge(model, x, y, model_kind, teacher_forcing_ratio)
            y_for_loss = y[..., :1] if model_kind in ("lstnn", "staeformer", "pdformer") else y
            loss = criterion(y_for_loss, y_hat)

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        m = calc_metrics(y_for_loss, y_hat, scaler=scaler_obj)
        bsz = x.size(0)
        seen += bsz
        meter["loss"]  += loss.item() * bsz
        meter["mae"]   += m["mae"] * bsz
        meter["rmse"]  += m["rmse"] * bsz
        meter["smape"] += m["smape"] * bsz

    for k in meter:
        meter[k] /= max(seen, 1)
    return meter

@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: HuberRegLoss,
    loader: DataLoader,
    device: torch.device,
    scaler_obj=None,
    model_kind: str = "tgcn",
) -> Dict[str, float]:
    model.eval()
    meter = {"loss": 0.0, "mae": 0.0, "rmse": 0.0, "smape": 0.0}
    seen = 0

    amp_enabled = (device.type == "cuda")
    autocast = torch.cuda.amp.autocast if amp_enabled else nullcontext

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with autocast(dtype=torch.float16):
            y_hat = _forward_bridge(model, x, y, model_kind, teacher_forcing_ratio=0.0)
            y_for_loss = y[..., :1] if model_kind in ("lstnn", "staeformer", "pdformer") else y
            loss = criterion(y_for_loss, y_hat)

        m = calc_metrics(y_for_loss, y_hat, scaler=scaler_obj)
        bsz = x.size(0)
        seen += bsz
        meter["loss"]  += loss.item() * bsz
        meter["mae"]   += m["mae"] * bsz
        meter["rmse"]  += m["rmse"] * bsz
        meter["smape"] += m["smape"] * bsz

    for k in meter:
        meter[k] /= max(seen, 1)
    return meter

# ----------------------------
# 入口：训练（支持 tgcn / stgcn / lstnn / staeformer / pdformer）
# ----------------------------
def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    adj: torch.Tensor,
    n_nodes: int,
    in_dim: int,
    seq_len: int,
    pred_len: int,
    hidden_dim: int = 64,
    out_dim: Optional[int] = None,
    dropout: float = 0.0,
    use_pred_for_next: bool = True,
    map_pred_to_infeat_dims: int = 1,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    scaler_obj=None,
    device: Optional[torch.device] = None,
    teacher_forcing_ratio: float = 0.0,
    best_ckpt_path: Optional[str] = None,
    last_ckpt_path: Optional[str] = None,
    resume_ckpt: Optional[str] = None,
    resume_states: bool = False,
    finetune_lr: Optional[float] = None,
    log_txt_path: Optional[str] = None,

    # ===== STGCN 超参 =====
    model_kind: str = "tgcn",                         # "tgcn" / "stgcn" / "lstnn" / "staeformer" / "pdformer"
    stgcn_graph_conv_type: str = "cheb_graph_conv",   # 'cheb_graph_conv' / 'graph_conv'
    stgcn_Kt: int = 3,
    stgcn_Ks: int = 3,
    stgcn_act: str = "glu",
    stgcn_droprate: float = 0.0,
    stgcn_blocks=None,

    # ===== LSTNN（MultiscaleMLP）超参 =====
    msmlp_patch_len: int = 24,
    msmlp_stride: int = 4,
    msmlp_td_size: int = 288,
    msmlp_dw_size: int = 7,
    msmlp_d_td: int = 8,
    msmlp_d_dw: int = 4,
    msmlp_d_d:  int = 32,
    msmlp_d_spa:int = 8,
    msmlp_if_time_in_day: bool = True,
    msmlp_if_day_in_week: bool = True,
    msmlp_if_spatial: bool = True,
    msmlp_num_layer: int = 2,

    # ===== STAEformer 超参 =====
    stf_steps_per_day: int = 288,
    stf_input_embedding_dim: int = 24,
    stf_tod_embedding_dim: int = 24,
    stf_dow_embedding_dim: int = 24,
    stf_spatial_embedding_dim: int = 0,
    stf_adaptive_embedding_dim: int = 80,
    stf_feed_forward_dim: int = 256,
    stf_num_heads: int = 4,
    stf_num_layers: int = 3,
    stf_dropout: float = 0.1,
    stf_use_mixed_proj: bool = True,

    # ===== PDFormer 超参（与 pdformer.py 对齐）=====
    pdf_embed_dim: int = 64,
    pdf_skip_dim: int = 128,
    pdf_lape_dim: int = 8,
    pdf_geo_heads: int = 4,
    pdf_sem_heads: int = 2,
    pdf_time_heads: int = 2,
    pdf_mlp_ratio: float = 4.0,
    pdf_s_attn_size: int = 3,
    pdf_t_attn_size: int = 3,
    pdf_enc_depth: int = 3,
    pdf_type_ln: str = "pre",
    pdf_short_path: str = "hop",       # "hop" or "dist"
    pdf_drop: float = 0.0,
    pdf_attn_drop: float = 0.0,
    pdf_drop_path: float = 0.1,
    pdf_far_mask_delta: float = 5.0,
    pdf_dtw_delta: int = 3,
    pdf_use_curriculum: bool = False,
    pdf_step_size: int = 2500,
    pdf_task_level: Optional[int] = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = out_dim or in_dim

    # ---- 日志文件（可选）----
    log_f = None
    if log_txt_path:
        headers = [
            "epoch",
            "train_loss", "train_mae", "train_rmse", "train_smape",
            "val_loss",   "val_mae",   "val_rmse",   "val_smape",
            "lr", "time_sec"
        ]
        log_f = _init_epoch_log(log_txt_path, headers)

    # ===== 构建模型 =====
    if model_kind == "tgcn":
        model = TGCNForecaster(
            adj=adj.to(device),
            num_nodes=n_nodes,
            input_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            horizon=pred_len,
            dropout=dropout,
            use_pred_for_next=use_pred_for_next,
            map_pred_to_infeat_dims=map_pred_to_infeat_dims,
        ).to(device)

    elif model_kind == "stgcn":
        if STGCNChebGraphConv is None:
            raise ImportError("未找到 STGCN 类。请确认已把 model/STGCN.py 放在工程根目录并可被导入。")
        if stgcn_blocks is None:
            end_channel = pred_len * out_dim
            stgcn_blocks = [
                [in_dim],
                [64, 16, 64],
                [64, 16, 64],
                [64, 128],
                [end_channel],
            ]
        gso = build_gso_for_stgcn(adj.to(device), graph_conv_type=stgcn_graph_conv_type, add_self_loops=True)
        args_ns = SimpleNamespace(
            n_his=seq_len,
            Kt=stgcn_Kt,
            Ks=stgcn_Ks,
            act_func=stgcn_act,
            graph_conv_type=stgcn_graph_conv_type,
            gso=gso,
            enable_bias=True,
            droprate=stgcn_droprate,
        )
        if stgcn_graph_conv_type == "cheb_graph_conv":
            model = STGCNChebGraphConv(args_ns, stgcn_blocks, n_nodes).to(device)
        else:
            model = STGCNGraphConv(args_ns, stgcn_blocks, n_nodes).to(device)

    elif model_kind == "lstnn":
        lstnn_args = dict(
            node_size=n_nodes,
            input_len=seq_len,
            input_dim=in_dim,
            output_len=pred_len,
            patch_len=msmlp_patch_len,
            stride=msmlp_stride,
            td_size=msmlp_td_size,
            dw_size=msmlp_dw_size,
            d_td=msmlp_d_td,
            d_dw=msmlp_d_dw,
            d_d=msmlp_d_d,
            d_spa=msmlp_d_spa,
            if_time_in_day=msmlp_if_time_in_day,
            if_day_in_week=msmlp_if_day_in_week,
            if_spatial=msmlp_if_spatial,
            num_layer=msmlp_num_layer,
        )
        model = MultiscaleMLP(**lstnn_args).to(device)

    elif model_kind == "staeformer":
        model = STAEformer(
            num_nodes=n_nodes,
            in_steps=seq_len,
            out_steps=pred_len,
            steps_per_day=stf_steps_per_day,
            input_dim=in_dim,
            output_dim=out_dim,
            input_embedding_dim=stf_input_embedding_dim,
            tod_embedding_dim=stf_tod_embedding_dim,
            dow_embedding_dim=stf_dow_embedding_dim,
            spatial_embedding_dim=stf_spatial_embedding_dim,
            adaptive_embedding_dim=stf_adaptive_embedding_dim,
            feed_forward_dim=stf_feed_forward_dim,
            num_heads=stf_num_heads,
            num_layers=stf_num_layers,
            dropout=stf_dropout,
            use_mixed_proj=stf_use_mixed_proj,
        ).to(device)

    elif model_kind == "pdformer":
        # --- 准备 data_feature 所需矩阵 ---
        A_np = adj.detach().cpu().numpy().astype(np.float32)
        # 最短路距离（权重>0 按可达）
        sd_mx = _compute_shortest_path_dist(A_np).astype(np.float32)
        # hop 距离
        sh_mx = _compute_hop_dist(A_np).astype(np.float32)
        # 用 hop 距离近似 DTW 相似度矩阵（越小越相近）
        dtw_matrix = sh_mx.copy().astype(np.float32)

        # pattern_keys: 形状 (Tin, N, s_attn_size, out_dim)
        pattern_keys = np.zeros((seq_len, n_nodes, pdf_s_attn_size, out_dim), dtype=np.float32)
        

        data_feature = dict(
            num_nodes=n_nodes,
            feature_dim=in_dim,
            ext_dim=0,
            num_batches=len(train_loader),
            dtw_matrix=dtw_matrix,
            adj_mx=None,
            sd_mx=sd_mx,
            sh_mx=sh_mx,
            pattern_keys=pattern_keys,
            scaler=None,
        )

        config = dict(
            embed_dim=pdf_embed_dim,
            skip_dim=pdf_skip_dim,
            lape_dim=pdf_lape_dim,
            geo_num_heads=pdf_geo_heads,
            sem_num_heads=pdf_sem_heads,
            t_num_heads=pdf_time_heads,
            mlp_ratio=pdf_mlp_ratio,
            qkv_bias=True,
            drop=pdf_drop,
            attn_drop=pdf_attn_drop,
            drop_path=pdf_drop_path,
            s_attn_size=pdf_s_attn_size,
            t_attn_size=pdf_t_attn_size,
            enc_depth=pdf_enc_depth,
            type_ln=pdf_type_ln,
            type_short_path=pdf_short_path,
            output_dim=out_dim,           # 通常=1（只预测值）
            input_window=seq_len,
            output_window=pred_len,
            add_time_in_day=False,
            add_day_in_week=False,
            device=device,
            world_size=1,
            huber_delta=1.0,
            quan_delta=0.25,
            far_mask_delta=pdf_far_mask_delta,
            dtw_delta=pdf_dtw_delta,
            use_curriculum_learning=pdf_use_curriculum,
            step_size=pdf_step_size,
            max_epoch=epochs,
            task_level=(pdf_task_level if pdf_task_level is not None else pred_len),
        )

        model = PDFormer(config, data_feature).to(device)

        # 计算拉普拉斯位置编码（(N, lape_dim)），供前向用
        lap_mx = _laplacian_positional_eigvecs(adj.to(device), pdf_lape_dim)  # (N, lape_dim)
        # 挂到 model 上（_forward_bridge 里读取）
        model._lap_mx = lap_mx

    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    _print_model_profile(
        model,
        model_kind=model_kind,
        n_nodes=n_nodes, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
        seq_len=seq_len, pred_len=pred_len, dropout=dropout
    )

    criterion = HuberRegLoss(delta=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    # ---- 断点续训 / 微调 ----
    start_epoch = 1
    best_val = float("inf")
    if resume_ckpt is not None and os.path.isfile(resume_ckpt):
        print(f"[Resume] Loading checkpoint from: {resume_ckpt}")
        start_epoch, best_val_loaded, _cfg = load_checkpoint(
            resume_ckpt, model,
            optimizer if resume_states else None,
            scheduler if resume_states else None,
            map_location=device
        )
        if (not resume_states) and (finetune_lr is not None):
            for g in optimizer.param_groups:
                g["lr"] = finetune_lr
            print(f"[Finetune] Set optimizer LR to {finetune_lr:g}")
        best_val = min(best_val, best_val_loaded)
        print(f"[Resume] Start from epoch {start_epoch}, best_val={best_val:.6f}")
    else:
        print("[Resume] No checkpoint. Train from scratch.")

    # 打印 AMP 状态
    if device.type == "cuda":
        print("[AMP] Enabled: autocast(fp16) + GradScaler")
    else:
        print("[AMP] Disabled (non-CUDA device)")

    best_state = None
    wait = 0
    end_epoch = start_epoch + epochs - 1

    try:
        for epoch in range(start_epoch, end_epoch + 1):
            t0 = time.time()
            tr = train_one_epoch(
                model, criterion, optimizer, train_loader, device, scaler_obj,
                grad_clip=1.0, teacher_forcing_ratio=teacher_forcing_ratio,
                model_kind=model_kind
            )
            va = evaluate(model, criterion, val_loader, device, scaler_obj, model_kind=model_kind)

            monitor_value = va["rmse"]
            prev_lrs = [g["lr"] for g in optimizer.param_groups]
            scheduler.step(monitor_value)
            curr_lrs = [g["lr"] for g in optimizer.param_groups]
            t1 = time.time()

            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"[Epoch {epoch:03d}] "
                  f"Train: loss={tr['loss']:.4f} MAE={tr['mae']:.4f} RMSE={tr['rmse']:.4f} sMAPE={tr['smape']:.4f} "
                  f"| Val: loss={va['loss']:.4f} MAE={va['mae']:.4f} RMSE={va['rmse']:.4f} sMAPE={va['smape']:.4f} "
                  f"| lr={cur_lr:.2e} | time={t1-t0:.1f}s")

            if any(c < p for p, c in zip(prev_lrs, curr_lrs)):
                print(f"[Scheduler] ReduceLROnPlateau: lr reduced -> {curr_lrs}")

            if log_f is not None:
                log_f.write(
                    f"{epoch}\t"
                    f"{tr['loss']:.6f}\t{tr['mae']:.6f}\t{tr['rmse']:.6f}\t{tr['smape']:.6f}\t"
                    f"{va['loss']:.6f}\t{va['mae']:.6f}\t{va['rmse']:.6f}\t{va['smape']:.6f}\t"
                    f"{cur_lr:.6g}\t{t1 - t0:.3f}\n"
                )

            # 保存 last（每轮覆盖）
            if last_ckpt_path:
                save_checkpoint(
                    last_ckpt_path, model,
                    optimizer if resume_states else None,
                    scheduler if resume_states else None,
                    epoch=epoch, monitor_value=monitor_value,
                    config=_pack_config(
                        model_kind=model_kind,
                        n_nodes=n_nodes, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                        seq_len=seq_len, pred_len=pred_len, dropout=dropout
                    )
                )

            # 保存 best
            if monitor_value < best_val - 1e-6:
                best_val = monitor_value
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                wait = 0
                if best_ckpt_path:
                    save_checkpoint(
                        best_ckpt_path, model, None, None,
                        epoch=epoch, monitor_value=best_val,
                        config=_pack_config(
                            model_kind=model_kind,
                            n_nodes=n_nodes, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                            seq_len=seq_len, pred_len=pred_len, dropout=dropout
                        )
                    )
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping.")
                    break
    finally:
        if log_f is not None:
            log_f.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
