#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STMAN v2 优化版 - METR-LA 训练脚本
修复: PE频率、解码器步长嵌入、零值掩码metrics、ReLU输出、LR patience
"""
import os, sys, argparse
import re
import numpy as np
import pandas as pd
import torch

os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONUNBUFFERED"] = "1"  # 实时输出日志
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import append_calendar_features, load_adjacency_csv, create_data_loaders
from train import train_model


def infer_resume_architecture(resume_ckpt: str):
    """Infer warm-start-critical architecture choices from a checkpoint."""
    if not resume_ckpt or (not os.path.isfile(resume_ckpt)):
        return {}
    try:
        ckpt = torch.load(resume_ckpt, map_location="cpu")
    except Exception as exc:
        print(f"[Resume] Failed to inspect checkpoint architecture from {resume_ckpt}: {exc}")
        return {}

    info = {}
    cfg = ckpt.get("config") or {}
    for key in (
        "dataset",
        "seq_len",
        "pred_len",
        "scaler",
        "spatial_mode",
        "k_hop",
        "direct_head_mode",
        "use_direct_pred",
        "direct_step_refine",
        "decoder_future_tod",
        "gated_residual",
        "scale_target_only",
        "calendar_features",
        "use_future_time_features",
        "time_feature_start",
    ):
        if key in cfg and cfg[key] is not None:
            info[key] = cfg[key]

    match = re.match(r"(.+?)_(12|48|96|288|864|2016)_.+_(best|last)\.pt$", os.path.basename(resume_ckpt))
    if match:
        info.setdefault("dataset", match.group(1))
        info.setdefault("pred_len", int(match.group(2)))

    model_state = ckpt.get("model") or {}
    hop_sizes = sorted({
        int(value.numel())
        for key, value in model_state.items()
        if key.endswith("hop_weights") and getattr(value, "ndim", None) == 1
    })
    if hop_sizes:
        info.setdefault("spatial_mode", "gcn")
        if len(hop_sizes) == 1 and hop_sizes[0] >= 1:
            info.setdefault("k_hop", hop_sizes[0] - 1)

    state_keys = list(model_state.keys())
    if any(key.endswith("out_proj_direct.weight") for key in state_keys):
        info.setdefault("use_direct_pred", True)
        info.setdefault("direct_head_mode", "linear")
    elif any(key.endswith("chunk_query.weight") or key.endswith("temporal_key.weight") for key in state_keys):
        info.setdefault("use_direct_pred", True)
        info.setdefault("direct_head_mode", "attn")
    elif any(key.endswith("temporal_chunk_proj.weight") for key in state_keys):
        info.setdefault("use_direct_pred", True)
        info.setdefault("direct_head_mode", "chunkwise")
    elif any(key.endswith("temporal_agg.weight") for key in state_keys) and any(
        key.endswith("chunk_emb.weight") for key in state_keys
    ):
        info.setdefault("use_direct_pred", True)
        info.setdefault("direct_head_mode", "legacy")
    if any("chunk_step_emb.weight" in key or "step_refine." in key for key in state_keys):
        info.setdefault("direct_step_refine", True)

    return info


def apply_resume_guard(args):
    """Disable unsafe model-only warm starts before entering train_model()."""
    if not args.resume_ckpt or (not os.path.isfile(args.resume_ckpt)):
        return
    if args.resume_mode != "model":
        return
    if args.allow_resume_arch_mismatch:
        print("[ResumeGuard] skip preflight because --allow_resume_arch_mismatch is set")
        return

    resume_arch = infer_resume_architecture(args.resume_ckpt)
    if not resume_arch:
        return

    reasons = []
    src_dataset = resume_arch.get("dataset")
    if src_dataset and str(src_dataset) != str(args.dataset):
        reasons.append(f"dataset mismatch: ckpt={src_dataset} target={args.dataset}")

    src_pred_len = resume_arch.get("pred_len")
    if src_pred_len is not None and int(src_pred_len) != int(args.pred_len):
        reasons.append(f"pred_len mismatch: ckpt={int(src_pred_len)} target={int(args.pred_len)}")

    src_scaler = resume_arch.get("scaler")
    if src_scaler and str(src_scaler) != str(args.scaler):
        reasons.append(f"scaler mismatch: ckpt={src_scaler} target={args.scaler}")

    src_use_direct = resume_arch.get("use_direct_pred")
    target_use_direct = bool(args.use_direct_pred)
    if src_use_direct is not None and bool(src_use_direct) != target_use_direct:
        reasons.append(
            "prediction family mismatch: "
            f"ckpt_use_direct={int(bool(src_use_direct))} target_use_direct={int(target_use_direct)}"
        )

    src_direct_mode = resume_arch.get("direct_head_mode")
    if target_use_direct and src_direct_mode and str(src_direct_mode) != str(args.direct_head_mode):
        reasons.append(
            f"direct_head_mode mismatch: ckpt={src_direct_mode} target={args.direct_head_mode}"
        )

    src_spatial_mode = resume_arch.get("spatial_mode")
    if src_spatial_mode and str(src_spatial_mode) != str(args.spatial_mode):
        reasons.append(
            f"spatial_mode mismatch: ckpt={src_spatial_mode} target={args.spatial_mode}"
        )

    src_k_hop = resume_arch.get("k_hop")
    if args.spatial_mode in ("gcn", "hybrid") and src_k_hop is not None and int(src_k_hop) != int(args.k_hop):
        reasons.append(f"k_hop mismatch: ckpt={int(src_k_hop)} target={int(args.k_hop)}")

    src_seq_len = resume_arch.get("seq_len")
    if (not target_use_direct) and src_seq_len is not None and int(src_seq_len) != int(args.seq_len):
        reasons.append(f"AR seq_len mismatch: ckpt={int(src_seq_len)} target={int(args.seq_len)}")

    src_decoder_future_tod = resume_arch.get("decoder_future_tod")
    target_decoder_future_tod = (not bool(args.no_decoder_future_tod))
    if ((not target_use_direct) and src_decoder_future_tod is not None and
            bool(src_decoder_future_tod) != bool(target_decoder_future_tod)):
        reasons.append(
            "decoder_future_tod mismatch: "
            f"ckpt={int(bool(src_decoder_future_tod))} target={int(bool(target_decoder_future_tod))}"
        )

    if not reasons:
        print(f"[ResumeGuard] preflight passed for {args.resume_ckpt}")
        return

    print(f"[ResumeGuard] disable resume and start fresh: {args.resume_ckpt}")
    for reason in reasons:
        print(f"[ResumeGuard] - {reason}")
    args.resume_ckpt = None
    args.seed_best_mae = None
    args.finetune_lr = None


def split_raw_data(raw: np.ndarray, split_rate: float = 0.6, val_ratio: float = 0.2):
    """Split raw series by explicit train and validation ratios."""
    split_rate = float(split_rate)
    val_ratio = float(val_ratio)
    if split_rate <= 0.0 or val_ratio < 0.0 or split_rate + val_ratio >= 1.0:
        raise ValueError(
            f"invalid split ratios: split_rate={split_rate:g}, val_ratio={val_ratio:g}"
        )
    total = raw.shape[0]
    n_train = int(total * split_rate)
    n_val = int(total * val_ratio)
    return raw[:n_train], raw[n_train:n_train + n_val], raw[n_train + n_val:]


def load_node_indices(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    frame = pd.read_csv(path)
    for column in ("matrix_index", "node_index", "index"):
        if column in frame.columns:
            values = frame[column].to_numpy(dtype=np.int64)
            break
    else:
        values = frame.iloc[:, 0].to_numpy(dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Invalid node index file: {path}")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="METR-LA")
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--enc_layers", type=int, default=3)
    parser.add_argument("--dec_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ff_multiplier", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_workers", type=int, default=4)
    parser.add_argument("--eval_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--split_rate", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--scaler", type=str, default="standard")
    parser.add_argument("--tag", type=str, default="v2")
    parser.add_argument("--progress_log", type=str, default=None,
                        help="独立进度文件路径，每 log_interval epoch 写一行 best_MAE")
    parser.add_argument("--log_interval", type=int, default=5,
                        help="每隔多少 epoch 写入进度文件（best 改善时额外写）")
    parser.add_argument("--use_direct_pred", action="store_true",
                        help="使用直接预测头（temporal_agg + Linear）替代自回归解码器，适合长预测步长")
    parser.add_argument("--direct_head_mode", type=str, default="legacy",
                        choices=["legacy", "chunkwise", "attn", "linear"],
                        help="直接预测头模式: legacy=旧式chunk-MLP, chunkwise=按未来chunk独立聚合, attn=按chunk查询自适应聚合历史, linear=旧版全horizon线性头")
    parser.add_argument("--direct_step_refine", type=int, default=0, choices=[0, 1],
                        help="为非线性 direct head 打开 chunk 内逐步 residual refiner，适合长步长 warm-start 微调")
    parser.add_argument("--spatial_mode", type=str, default="attention",
                        choices=["attention", "gcn", "hybrid"],
                        help="空间混合模式: attention=密集注意力, gcn=多跳图卷积, hybrid=多跳图卷积+分形注意力融合")
    parser.add_argument("--k_hop", type=int, default=8,
                        help="GCN模式下的多跳数(默认8), 决定每个节点能看到多远的邻居")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau"],
                        help="LR调度器: cosine=CosineAnnealingWarmRestarts, plateau=ReduceLROnPlateau")
    parser.add_argument("--plateau_patience", type=int, default=8,
                        help="ReduceLROnPlateau patience；平台期修复任务可降到3-4")
    parser.add_argument("--eta_min", type=float, default=1e-5,
                        help="cosine scheduler 最小学习率")
    parser.add_argument("--cosine_T0", type=int, default=30,
                        help="cosine warm restart 首个周期长度（epoch数）")
    parser.add_argument("--cosine_T_mult", type=int, default=2,
                        help="cosine周期倍增因子: 1=固定周期, 2=递增(30→60→120)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="AdamW weight decay (过拟合任务可加大到1e-4)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="多卡并行GPU ID，逗号分隔，例如 '0,1' 或 '1,2'")
    parser.add_argument("--amp_policy", type=str, default="auto",
                        choices=["auto", "bf16", "fp16", "off"],
                        help="AMP精度策略: auto=自动选bf16/fp16, bf16, fp16(+GradScaler), off=禁用AMP")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="恢复训练的checkpoint路径，传last.pt即可从断点继续")
    parser.add_argument("--resume_mode", type=str, default="states",
                        choices=["states", "model"],
                        help="恢复模式: states=加载模型+optimizer+scheduler, model=仅加载模型权重")
    parser.add_argument("--finetune_lr", type=float, default=None,
                        help="仅在 resume_mode=model 时生效: 使用新的学习率继续训练")
    parser.add_argument("--seed_best_mae", type=float, default=None,
                        help="恢复 best-ckpt 时显式播种历史 best_MAE，避免旧 checkpoint 的非 MAE monitor 干扰早停/监控")
    parser.add_argument("--allow_resume_arch_mismatch", action="store_true",
                        help="允许 warm start checkpoint 与当前 spatial/direct head 配置不一致（默认自动对齐）")
    parser.add_argument("--diversity_weight", type=float, default=0.0,
                        help="防坍缩时间多样性损失权重 (推荐0.1, 0=禁用)")
    parser.add_argument("--diversity_threshold", type=float, default=0.3,
                        help="pred_std/true_std 最低比例阈值")
    parser.add_argument("--vmg_weight", type=float, default=0.0,
                        help="variance matching loss 权重 (推荐0.3, 让预测波动匹配真实波动)")
    parser.add_argument("--qml_weight", type=float, default=0.0,
                        help="quantile matching loss 权重 (推荐0.3)")
    parser.add_argument("--horizon_weight_mode", type=str, default="uniform",
                        choices=["uniform", "linear", "exp", "final"],
                        help="未来步长损失加权: uniform=旧逻辑, linear/exp=越远权重越高, final=只增强最后一步")
    parser.add_argument("--horizon_last_weight", type=float, default=1.0,
                        help="最后一步相对第一步的损失权重；内部会归一化均值，默认1保持旧逻辑")
    parser.add_argument("--delta_weight", type=float, default=0.0,
                        help="趋势/增量匹配损失权重，用于减少过平滑和峰谷滞后")
    parser.add_argument("--monitor_mode", type=str, default="mae",
                        choices=["mae", "strict"],
                        help="best checkpoint 选择口径: mae=纯MAE, strict=MAE+quality gate 惩罚")
    parser.add_argument("--strict_diversity_gate", type=float, default=0.5,
                        help="strict monitor 的 diversity 门槛")
    parser.add_argument("--strict_vs_persistence_gate", type=float, default=0.95,
                        help="strict monitor 的 vs_persistence 门槛")
    parser.add_argument("--no_residual_delta", action="store_true",
                        help="禁用残差修正，强制模型预测绝对值(防止半稳态数据如METR-LA坍缩到预测last_val)")
    parser.add_argument("--no_decoder_future_tod", action="store_true",
                        help="自回归解码时不注入未来TOD；兼容旧 autoregressive checkpoint")
    parser.add_argument("--gated_residual", action="store_true",
                        help="使用可学习 residual gate 替代固定 last-value 残差")
    parser.add_argument("--add_dow_feature", action="store_true",
                        help="在 TOD 后追加 day-of-week 原始日历通道")
    parser.add_argument("--sf_scale", type=float, default=1.0,
                        help="scale sensitivity: multiply spatial fractal vectors")
    parser.add_argument("--tf_scale", type=float, default=1.0,
                        help="scale sensitivity: multiply temporal fractal vectors")
    parser.add_argument("--scale_target_only", action="store_true",
                        help="只标准化目标通道，TOD/DOW 保持 [0,1) 原始日历尺度")
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="参数EMA衰减率；>0时启用EMA收尾评估")
    parser.add_argument("--ema_start_epoch", type=int, default=0,
                        help="从哪个epoch开始更新EMA")
    parser.add_argument("--ema_eval_interval", type=int, default=1,
                        help="每隔多少epoch评估一次EMA shadow")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="梯度累积步数；batch=1/2 时提升有效 batch，降低验证曲线抖动")
    parser.add_argument("--head_warmup_epochs", type=int, default=0,
                        help="warm-start 新预测头时先只训练预测头若干 epoch，再解冻全模型")
    parser.add_argument("--head_lr_mult", type=float, default=1.0,
                        help="预测头参数组学习率倍率；配合 head_warmup_epochs 使用")
    parser.add_argument("--roi_stop_epochs", type=int, default=0,
                        help="连续若干次验证没有达到最小MAE下降则提前停止；0=禁用")
    parser.add_argument("--roi_min_mae_drop", type=float, default=0.0,
                        help="roi_stop_epochs窗口内要求达到的最小MAE下降")
    parser.add_argument("--node_indices_csv", type=str, default=None,
                        help="可选：子图节点索引CSV；用于表4地理尺度实验")
    parser.add_argument("--adj_path", type=str, default=None,
                        help="可选：子图邻接矩阵CSV；默认使用 Datasets/<dataset>/adj_matrix.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="checkpoint输出目录；默认保持旧行为写入 ./checkpoints")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    # 多卡并行解析
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = args.dataset
    data_dir = f"Datasets/{ds}"
    node_indices = load_node_indices(args.node_indices_csv)

    # 邻接矩阵
    adj_path = args.adj_path or os.path.join(data_dir, "adj_matrix.csv")
    A = load_adjacency_csv(adj_path,
                           normalize="sym", add_self_loops=True).to(device)
    if node_indices is not None and A.shape[0] != len(node_indices):
        idx_t = torch.as_tensor(node_indices, dtype=torch.long, device=device)
        A = A.index_select(0, idx_t).index_select(1, idx_t)
    N = A.shape[0]

    # 分形特征
    SF_np = np.load(f"spatial_fractal_vectors_{ds}.npy")
    TF_np = np.load(f"train_timefractals_{ds}.npy")
    if node_indices is not None and SF_np.shape[0] != len(node_indices):
        SF_np = SF_np[node_indices]
    if node_indices is not None and TF_np.shape[0] != len(node_indices):
        TF_np = TF_np[node_indices]
    SF = torch.tensor(SF_np, dtype=torch.float32).to(device)
    TF = torch.tensor(TF_np, dtype=torch.float32).to(device)
    if args.sf_scale != 1.0 or args.tf_scale != 1.0:
        SF = SF * float(args.sf_scale)
        TF = TF * float(args.tf_scale)
        print(f"Applied fractal scale sensitivity: sf_scale={args.sf_scale} tf_scale={args.tf_scale}")

    # 加载数据
    csv_path = os.path.join(data_dir, f"{ds}.csv")
    h5_path = os.path.join(data_dir, "metr-la.h5")
    npz_path = os.path.join(data_dir, f"{ds}.npz")
    # 优先用NPZ（保留多通道），其次H5，最后CSV
    if os.path.exists(npz_path):
        raw = np.load(npz_path)["data"].astype(np.float32)
        print(f"Loaded NPZ: {raw.shape}")
    elif os.path.exists(h5_path):
        raw = pd.read_hdf(h5_path).values.astype(np.float32)
    elif os.path.exists(csv_path):
        raw = pd.read_csv(csv_path, header=None).values.astype(np.float32)
    else:
        raise FileNotFoundError(f"No data found for {ds}")

    if raw.ndim == 2:
        raw = raw[:, :, np.newaxis]  # [T, N, 1]
    if node_indices is not None and raw.shape[1] != len(node_indices):
        raw = raw[:, node_indices, :]
    T_total = raw.shape[0]

    raw = append_calendar_features(raw, steps_per_day=288, add_dow=args.add_dow_feature)
    calendar_features = "tod+dow" if args.add_dow_feature else "tod"
    print(f"Added calendar features ({calendar_features}): raw shape = {raw.shape}")

    X_train, X_val, X_test = split_raw_data(
        raw, split_rate=args.split_rate, val_ratio=args.val_ratio
    )
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    print(f"Data: {raw.shape}, Split: train={n_train}, val={n_val}, test={X_test.shape[0]}")

    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        X_train, X_val, X_test,
        seq_len=args.seq_len, pred_len=args.pred_len,
        batch_size=args.batch_size, scaler_type=args.scaler,
        train_workers=args.train_workers, eval_workers=args.eval_workers,
        scale_target_only=args.scale_target_only, target_dim=1,
    )

    # checkpoint路径
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, f"{ds}_{args.pred_len}_{args.tag}_best.pt")
    last_path = os.path.join(ckpt_dir, f"{ds}_{args.pred_len}_{args.tag}_last.pt")

    print(f"\n{'='*60}")
    print(f"Training STMAN v2 on {ds} | seq={args.seq_len} pred={args.pred_len}")
    print(f"d_model={args.d_model} heads={args.n_heads} enc={args.enc_layers} dec={args.dec_layers} "
          f"ff={args.ff_multiplier} dropout={args.dropout}")
    print(
        f"scaler={args.scaler} split={args.split_rate} val_ratio={args.val_ratio} "
        f"lr={args.lr} patience={args.patience}"
    )
    print(f"Best checkpoint: {best_path}")
    print(f"{'='*60}\n")

    apply_resume_guard(args)

    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        A=A, SF=SF, TF=TF,
        n_nodes=N,
        in_dim=raw.shape[-1],
        out_dim=1,             # predict speed only
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        sf_dim=SF.shape[1],
        tf_dim=TF.shape[1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dropout=args.dropout,
        ff_multiplier=args.ff_multiplier,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        scaler_obj=scaler,
        device=device,
        best_ckpt_path=best_path,
        last_ckpt_path=last_path,
        use_amp=(args.amp_policy != "off"),
        amp_policy=args.amp_policy,
        allow_tf32=True,
        progress_log=args.progress_log,
        log_interval=args.log_interval,
        use_direct_pred=args.use_direct_pred,
        direct_head_mode=args.direct_head_mode,
        direct_step_refine=bool(args.direct_step_refine),
        spatial_mode=args.spatial_mode,
        k_hop=args.k_hop,
        scheduler_type=args.scheduler,
        plateau_patience=args.plateau_patience,
        eta_min=args.eta_min,
        cosine_T0=args.cosine_T0,
        cosine_T_mult=args.cosine_T_mult,
        diversity_weight=args.diversity_weight,
        diversity_threshold=args.diversity_threshold,
        vmg_weight=args.vmg_weight,
        qml_weight=args.qml_weight,
        horizon_weight_mode=args.horizon_weight_mode,
        horizon_last_weight=args.horizon_last_weight,
        delta_weight=args.delta_weight,
        monitor_mode=args.monitor_mode,
        strict_diversity_gate=args.strict_diversity_gate,
        strict_vs_persistence_gate=args.strict_vs_persistence_gate,
        residual_delta=(not args.no_residual_delta),
        decoder_future_tod=(not args.no_decoder_future_tod),
        gated_residual=args.gated_residual,
        scale_target_only=args.scale_target_only,
        calendar_features=calendar_features,
        ema_decay=args.ema_decay,
        ema_start_epoch=args.ema_start_epoch,
        ema_eval_interval=args.ema_eval_interval,
        grad_accum_steps=args.grad_accum_steps,
        head_warmup_epochs=args.head_warmup_epochs,
        head_lr_mult=args.head_lr_mult,
        roi_stop_epochs=args.roi_stop_epochs,
        roi_min_mae_drop=args.roi_min_mae_drop,
        allow_resume_mismatch=args.allow_resume_arch_mismatch,
        gpu_ids=gpu_ids,
        scaler_type=args.scaler,
        split_rate=args.split_rate,
        val_ratio=args.val_ratio,
        resume_ckpt=args.resume_ckpt,
        resume_states=(True if (args.resume_ckpt and args.resume_mode == "states") else False),
        finetune_lr=args.finetune_lr,
        resume_best_mae=args.seed_best_mae,
    )

if __name__ == "__main__":
    main()
