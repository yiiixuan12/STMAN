# train.py
# -*- coding: utf-8 -*-
import time
from typing import Optional, Dict
import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import STFractalTransformer
from metric import STFractalLoss


# ----------------------------
# Multi-GPU wrapper
# ----------------------------
class _MultiGPUWrapper(nn.Module):
    """将图数据 (A, SF, TF) 存为 buffer，使 DataParallel 自动复制到每张卡。
    forward 只接收可按 batch 拆分的 x 和 y_in。"""
    def __init__(self, model: nn.Module, A: torch.Tensor, SF: torch.Tensor, TF: torch.Tensor):
        super().__init__()
        self.inner = model
        self.register_buffer("A", A)
        self.register_buffer("SF", SF)
        self.register_buffer("TF", TF)

    def forward(self, x, y_in=None):
        return self.inner(x, self.A, self.SF, self.TF, y_in=y_in)


def _is_multigpu(model: nn.Module) -> bool:
    """检查模型是否被 DataParallel 包裹。"""
    return isinstance(model, nn.DataParallel)


def _get_inner_model(model: nn.Module) -> nn.Module:
    """获取内部原始模型（穿透 DataParallel + Wrapper）。"""
    if isinstance(model, nn.DataParallel):
        model = model.module
    if isinstance(model, _MultiGPUWrapper):
        model = model.inner
    return model


def _strip_parallel_prefix(name: str) -> str:
    for prefix in ("module.inner.", "module.", "inner."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _is_prediction_head_param(name: str) -> bool:
    name = _strip_parallel_prefix(name)
    head_prefixes = (
        "out_hidden",
        "out_proj",
        "out_proj_direct",
        "temporal_agg",
        "temporal_chunk_proj",
        "chunk_query",
        "temporal_key",
        "chunk_emb",
        "future_time_proj",
        "chunk_mlp",
        "chunk_step_emb",
        "future_step_time_proj",
        "step_refine",
        "dec_step_emb",
        "decoder",
    )
    return name.startswith(head_prefixes)


def _set_head_bootstrap_trainable(model: nn.Module, enabled: bool) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = (_is_prediction_head_param(name) if enabled else True)


def _make_optimizer(model: nn.Module, lr: float, weight_decay: float, head_lr_mult: float):
    if head_lr_mult <= 0:
        raise ValueError("head_lr_mult must be positive")
    if abs(head_lr_mult - 1.0) < 1e-12:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if _is_prediction_head_param(name):
            head_params.append(param)
        else:
            backbone_params.append(param)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr, "name": "backbone"})
    if head_params:
        groups.append({"params": head_params, "lr": lr * head_lr_mult, "name": "head"})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


# ----------------------------
# small utils
# ----------------------------
def _count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def _print_model_profile(model: nn.Module, **cfg):
    total, trainable = _count_parameters(model)
    print("=" * 60)
    print("Model: STFractalTransformer")
    if cfg:
        print("Config: " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
    print(f"Parameters: total={total:,} | trainable={trainable:,} (~{trainable/1e6:.2f} M)")
    approx_mb = trainable * 4 / 1024 / 1024
    print(f"Approx. model size: {approx_mb:.2f} MB (fp32)")
    print("=" * 60)

def inverse_transform_batch(arr: torch.Tensor, scaler=None, channel_idx: int = None) -> np.ndarray:
    """
    arr: [B, T, N, F] torch.Tensor
    scaler: utils.StandardScaler / MinMaxScaler，或 None
    channel_idx: if not None, only inverse-transform using that channel's stats
                 (for when arr has fewer channels than scaler was fitted on)
    return: numpy ndarray (同形状)
    """
    arr_np = arr.detach().float().cpu().numpy()  # float() handles bf16→float32 conversion
    if scaler is None:
        return arr_np
    B, T, N, F = arr_np.shape
    flat = arr_np.reshape(B * T, N, F)
    if channel_idx is not None and F == 1:
        # arr has 1 channel but scaler was fit on multi-channel data
        # extract the relevant channel's stats for inverse transform
        if hasattr(scaler, 'mean') and scaler.mean is not None:
            # StandardScaler
            s_mean = scaler.mean
            s_std = scaler.std
            if s_mean.ndim == 3:
                s_mean = s_mean[:, :, channel_idx:channel_idx+1]
                s_std = s_std[:, :, channel_idx:channel_idx+1]
            elif s_mean.ndim == 2:
                s_mean = s_mean[:, channel_idx:channel_idx+1]
                s_std = s_std[:, channel_idx:channel_idx+1]
            inv = flat * s_std + s_mean
        elif hasattr(scaler, 'min_val') and scaler.min_val is not None:
            # MinMaxScaler
            s_min = scaler.min_val
            s_scale = scaler.scale
            if s_min.ndim == 3:
                s_min = s_min[:, :, channel_idx:channel_idx+1]
                s_scale = s_scale[:, :, channel_idx:channel_idx+1]
            elif s_min.ndim == 2:
                s_min = s_min[:, channel_idx:channel_idx+1]
                s_scale = s_scale[:, channel_idx:channel_idx+1]
            inv = flat * s_scale + s_min
        else:
            inv = flat
    else:
        inv = scaler.inverse_transform(flat)
    return inv.reshape(B, T, N, F)


# ----------------------------
# helpers: save / load checkpoint
# ----------------------------
def _pack_config(**kwargs):
    return dict(**kwargs)


def compute_monitor_value(metrics: Dict[str, float],
                          mode: str = "mae",
                          strict_diversity_gate: float = 0.5,
                          strict_vs_persistence_gate: float = 0.95) -> float:
    mae = float(metrics["mae"])
    if mode == "mae":
        return mae
    if mode == "strict":
        diversity = float(metrics.get("diversity", 0.0))
        vs_persistence = float(metrics.get("vs_persistence", float("inf")))
        div_penalty = max(1.0, strict_diversity_gate / max(diversity, 1e-8))
        vsp_penalty = max(1.0, vs_persistence / max(strict_vs_persistence_gate, 1e-8))
        return mae * div_penalty * vsp_penalty
    raise ValueError(f"Unknown monitor mode: {mode}")

def save_checkpoint(path: str,
                    model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                    epoch: int,
                    monitor_value: float,
                    config: Dict,
                    state_override: Optional[Dict[str, torch.Tensor]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": (
            {k: v.detach().cpu() for k, v in state_override.items()}
            if state_override is not None
            else {k: v.detach().cpu() for k, v in model.state_dict().items()}
        ),
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

def save_named_checkpoint(kind: str,
                          path: str,
                          model: nn.Module,
                          optimizer: Optional[torch.optim.Optimizer],
                          scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                          epoch: int,
                          monitor_value: float,
                          config: Dict,
                          state_override: Optional[Dict[str, torch.Tensor]] = None):
    include_training_state = (kind == "last")
    save_checkpoint(
        path, model,
        optimizer if include_training_state else None,
        scheduler if include_training_state else None,
        epoch=epoch,
        monitor_value=monitor_value,
        config=config,
        state_override=state_override,
    )


class _ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        state = model.state_dict()
        for key, value in state.items():
            value_cpu = value.detach().cpu()
            if key not in self.shadow:
                self.shadow[key] = value_cpu.clone()
                continue
            shadow = self.shadow[key]
            if (not value_cpu.is_floating_point()) or (not shadow.is_floating_point()):
                self.shadow[key] = value_cpu.clone()
                continue
            shadow.mul_(self.decay).add_(value_cpu, alpha=1.0 - self.decay)

    def clone_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().clone()
            for key, value in self.shadow.items()
        }

def load_checkpoint(path: str,
                    model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                    map_location: Optional[torch.device] = None,
                    strict: bool = True,
                    guard_model_only_mismatch: bool = False,
                    max_missing_keys_model_only: int = 12,
                    max_shape_skips_model_only: int = 6):
    ckpt = torch.load(path, map_location=map_location or "cpu")
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as e:
        if strict:
            raise RuntimeError(f"strict checkpoint load failed for {path}: {e}") from e
        print(f"[Resume] strict load failed, fallback to non-strict: {e}")
        current_state = model.state_dict()
        filtered_state = {}
        skipped_keys = []
        for key, value in ckpt["model"].items():
            if key not in current_state:
                continue
            if getattr(current_state[key], "shape", None) != getattr(value, "shape", None):
                skipped_keys.append(
                    f"{key}:{tuple(value.shape)}->{tuple(current_state[key].shape)}"
                )
                continue
            filtered_state[key] = value
        if not filtered_state:
            raise RuntimeError(
                f"non-strict checkpoint load failed for {path}: no compatible tensors after filtering"
            ) from e
        incompatible = model.load_state_dict(filtered_state, strict=False)
        print(
            "[Resume] filtered non-strict load: "
            f"loaded={len(filtered_state)} "
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)} "
            f"shape_skipped={len(skipped_keys)}"
        )
        if skipped_keys:
            preview = ", ".join(skipped_keys[:8])
            if len(skipped_keys) > 8:
                preview += ", ..."
            print(f"[Resume] shape-skipped keys: {preview}")
        if guard_model_only_mismatch:
            if len(incompatible.missing_keys) > max_missing_keys_model_only:
                raise RuntimeError(
                    f"unsafe_model_only_resume: missing_keys={len(incompatible.missing_keys)} "
                    f"> {max_missing_keys_model_only}"
                )
            if len(skipped_keys) > max_shape_skips_model_only:
                raise RuntimeError(
                    f"unsafe_model_only_resume: shape_skipped={len(skipped_keys)} "
                    f"> {max_shape_skips_model_only}"
                )
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[Resume] optimizer state load failed, start optimizer fresh: {e}")
    if scheduler is not None and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"[Resume] scheduler state load failed, start scheduler fresh: {e}")
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val = ckpt.get("val_loss", float("inf"))
    cfg = ckpt.get("config", {})
    return start_epoch, best_val, cfg


def infer_progress_log_from_checkpoint(ckpt_path: str,
                                       workdir: Optional[str] = None) -> Optional[str]:
    bn = os.path.basename(ckpt_path or "")
    match = re.match(r"(.+?)_(12|48|96|288|864|2016)_(.+)_(best|last)\.pt$", bn)
    if not match:
        return None
    ds, pred_len, tag, _ = match.groups()
    root = workdir or os.path.dirname(os.path.abspath(__file__))
    ds_lower = ds.lower().replace("-", "")
    name = f"{ds_lower}_{pred_len}_{tag}_progress.log"
    candidates = [
        os.path.join(root, "logs", name),
        os.path.abspath(os.path.join(root, "..", "..", "results", "stman_logs", name)),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def read_seed_best_mae(progress_log: Optional[str]) -> Optional[float]:
    if not progress_log or (not os.path.isfile(progress_log)):
        return None
    try:
        with open(progress_log, "r", encoding="utf-8", errors="ignore") as fh:
            lines = [line.strip() for line in fh if line.strip()]
    except OSError:
        return None
    for line in reversed(lines):
        for token in line.split():
            if token.startswith("best_MAE="):
                try:
                    return float(token.split("=", 1)[1])
                except ValueError:
                    return None
    return None


def resolve_resume_checkpoint_policy(resume_ckpt: Optional[str],
                                     resume_states: bool,
                                     resume_best_mae: Optional[float] = None,
                                     workdir: Optional[str] = None) -> Dict[str, Optional[float]]:
    plan = {
        "path": resume_ckpt,
        "best_mae_seed": float(resume_best_mae) if resume_best_mae is not None else None,
        "source_best_mae": None,
        "reason": None,
    }
    if not resume_ckpt or resume_states or (not os.path.isfile(resume_ckpt)):
        return plan

    source_progress_log = infer_progress_log_from_checkpoint(resume_ckpt, workdir=workdir)
    source_best_mae = read_seed_best_mae(source_progress_log)
    if source_best_mae is not None:
        plan["source_best_mae"] = source_best_mae
        if plan["best_mae_seed"] is None:
            plan["best_mae_seed"] = source_best_mae
        else:
            plan["best_mae_seed"] = min(plan["best_mae_seed"], source_best_mae)

    if not resume_ckpt.endswith("_best.pt"):
        return plan

    # An explicit seed_best_mae means the caller intentionally chose this checkpoint
    # as a model-only warm start. Do not silently swap it to sibling *_last.pt.
    if resume_best_mae is not None:
        return plan

    sibling_last = resume_ckpt[:-8] + "_last.pt"
    if not os.path.isfile(sibling_last) or source_best_mae is None:
        return plan

    try:
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        loaded_val_loss = ckpt.get("val_loss")
    except Exception:
        return plan

    if loaded_val_loss is None:
        return plan

    try:
        loaded_val_loss = float(loaded_val_loss)
    except (TypeError, ValueError):
        return plan

    if loaded_val_loss > source_best_mae * 1.05:
        plan["path"] = sibling_last
        plan["reason"] = "prefer_last_over_loss_selected_best"
    return plan


def resolve_resume_monitor_state(best_val_loaded: float,
                                 resume_states: bool,
                                 resume_best_mae: Optional[float] = None):
    best_val = float(best_val_loaded) if (resume_states and np.isfinite(best_val_loaded)) else float("inf")
    best_mae = best_val if np.isfinite(best_val) else float("inf")
    if resume_best_mae is not None:
        seed = float(resume_best_mae)
        best_mae = min(best_mae, seed)
        best_val = min(best_val, seed)
    return best_val, best_mae


# ----------------------------
# metrics
# ----------------------------
def calc_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, scaler=None, eps: float = 1e-6, mask_value: float = 0.0, channel_idx: int = None) -> Dict[str, float]:
    """
    计算 MAE / RMSE / MAPE（可选反标准化，排除零值/缺失值）
    mask_value: 真实值等于此值的点被排除（标准交通预测做法，排除传感器故障点）
    channel_idx: if not None, inverse-transform using that channel's scaler stats
    """
    yt = inverse_transform_batch(y_true, scaler, channel_idx=channel_idx)
    yp = inverse_transform_batch(y_pred, scaler, channel_idx=channel_idx)

    # 零值掩码：排除传感器故障导致的零/极小速度记录
    # 使用阈值 1e-4 (inverse后的原始空间) 排除真零值
    mask = np.abs(yt) > 1e-4
    if mask.sum() == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    diff = yp - yt
    mae = float(np.mean(np.abs(diff[mask])))
    rmse = float(np.sqrt(np.mean(diff[mask] ** 2)))

    # 标准 MAPE：分母使用较大阈值避免极小值放大误差
    mape = float(np.mean(np.abs(diff[mask]) / np.maximum(np.abs(yt[mask]), 1.0)))
    return {"mae": mae, "rmse": rmse, "mape": mape}


def calc_quality_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    x_last: torch.Tensor,
    scaler=None,
    channel_idx: int = None,
) -> Dict[str, float]:
    """在原始尺度上计算真实性指标，避免 MAE 低但预测坍缩。"""
    yt = inverse_transform_batch(y_true, scaler, channel_idx=channel_idx)
    yp = inverse_transform_batch(y_pred, scaler, channel_idx=channel_idx)
    xl = inverse_transform_batch(x_last, scaler, channel_idx=channel_idx)

    mask = np.abs(yt) > 1e-4
    if mask.sum() == 0:
        return {
            "pred_std": 0.0,
            "true_std": 0.0,
            "diversity": 0.0,
            "persistence_mae": 0.0,
            "vs_persistence": 0.0,
            "acc10": 0.0,
            "acc15": 0.0,
        }

    diff = yp - yt
    rel_err = np.abs(diff[mask]) / np.maximum(np.abs(yt[mask]), 1.0)
    pred_std = float(np.mean(np.std(yp, axis=1)))
    true_std = float(np.mean(np.std(yt, axis=1)))
    diversity = pred_std / max(true_std, 1e-8)

    persistence_pred = np.broadcast_to(xl, yt.shape)
    persistence_mae = float(np.mean(np.abs(persistence_pred[mask] - yt[mask])))
    mae = float(np.mean(np.abs(diff[mask])))
    vs_persistence = mae / max(persistence_mae, 1e-8)

    return {
        "pred_std": pred_std,
        "true_std": true_std,
        "diversity": diversity,
        "persistence_mae": persistence_mae,
        "vs_persistence": vs_persistence,
        "acc10": float(np.mean(rel_err <= 0.10)),
        "acc15": float(np.mean(rel_err <= 0.15)),
    }


# ----------------------------
# AMP resolver
# ----------------------------
def _resolve_amp(device: torch.device, use_amp: bool, amp_policy: str):
    """
    返回 (amp_enabled, amp_dtype, scaler)
    amp_policy: 'auto' | 'bf16' | 'fp16' | 'off'
    """
    if device.type != "cuda" or not use_amp or amp_policy == "off":
        return False, torch.float32, None

    if amp_policy == "bf16" or (amp_policy == "auto" and torch.cuda.is_bf16_supported()):
        # BF16：不需要 GradScaler
        return True, torch.bfloat16, torch.cuda.amp.GradScaler(enabled=False)
    else:
        # FP16：需要 GradScaler
        return True, torch.float16, torch.cuda.amp.GradScaler(enabled=True)


# ----------------------------
# train / eval (AMP ready)
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    criterion: STFractalLoss,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    A: torch.Tensor,
    SF: torch.Tensor,
    TF: torch.Tensor,
    scaler_obj=None,             # 用于反标准化显示指标
    grad_clip: Optional[float] = 1.0,
    teacher_forcing: bool = False,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    out_dim: int = None,         # if model outputs fewer channels than input
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    model.train()
    meter = {"loss": 0.0, "pred": 0.0, "sparse": 0.0, "var": 0.0, "sym": 0.0, "vmg": 0.0, "qml": 0.0, "delta": 0.0}
    seen = 0
    grad_accum_steps = max(int(grad_accum_steps), 1)

    multigpu = _is_multigpu(model)

    # 常量放 device（避免每 batch 复制）；多卡模式下 A/SF/TF 已在 wrapper buffer 中
    if not multigpu:
        A = A.to(device)
        SF = SF.to(device)
        TF = TF.to(device)

    autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype)

    _out_dim = out_dim  # number of output channels (may differ from input)
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(loader)
    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device)  # [B, T_in, N, F]
        y = y.to(device)  # [B, T_out, N, F]
        y_in = y if teacher_forcing else None

        with autocast_ctx:
            if multigpu:
                y_hat, aux = model(x, y_in=y_in)
            else:
                y_hat, aux = model(x, A, SF, TF, y_in=y_in)
            # Slice target to match model output channels
            y_target = y[:, :, :, :_out_dim] if (_out_dim is not None and _out_dim < y.shape[-1]) else y
            loss, parts = criterion(y_true=y_target, y_pred=y_hat, aux=aux)

        scaled_loss = loss / grad_accum_steps
        should_step = (batch_idx % grad_accum_steps == 0) or (batch_idx == total_batches)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
            if should_step:
                # 先unscale再clip，防止缩放影响裁剪
                scaler.unscale_(optimizer)
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            scaled_loss.backward()
            if should_step:
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        bsz = x.size(0)
        seen += bsz
        meter["loss"]  += float(loss.item()) * bsz
        meter["pred"]  += float(parts["pred"].item()) * bsz
        meter["sparse"]+= float(parts["sparse"].item()) * bsz
        meter["var"]   += float(parts["var"].item()) * bsz
        meter["sym"]   += float(parts["sym"].item()) * bsz
        meter["vmg"]   += float(parts.get("vmg", torch.tensor(0.0)).item()) * bsz
        meter["qml"]   += float(parts.get("qml", torch.tensor(0.0)).item()) * bsz
        meter["delta"] += float(parts.get("delta", torch.tensor(0.0)).item()) * bsz

    for k in meter:
        meter[k] /= max(seen, 1)
    return meter


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: STFractalLoss,
    loader: DataLoader,
    device: torch.device,
    A: torch.Tensor,
    SF: torch.Tensor,
    TF: torch.Tensor,
    scaler_obj=None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    out_dim: int = None,
) -> Dict[str, float]:
    model.eval()
    meter = {"loss": 0.0, "pred": 0.0, "sparse": 0.0, "var": 0.0, "sym": 0.0, "vmg": 0.0, "qml": 0.0, "delta": 0.0,
             "mae": 0.0, "rmse": 0.0, "mape": 0.0,
             "pred_std": 0.0, "true_std": 0.0, "persistence_mae": 0.0,
             "acc10": 0.0, "acc15": 0.0}
    seen = 0

    multigpu = _is_multigpu(model)
    if not multigpu:
        A = A.to(device)
        SF = SF.to(device)
        TF = TF.to(device)
    _out_dim = out_dim

    autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with autocast_ctx:
            if multigpu:
                y_hat, aux = model(x, y_in=None)
            else:
                y_hat, aux = model(x, A, SF, TF, y_in=None)
            y_target = y[:, :, :, :_out_dim] if (_out_dim is not None and _out_dim < y.shape[-1]) else y
            loss, parts = criterion(y_true=y_target, y_pred=y_hat, aux=aux)

        _ch_idx = 0 if (_out_dim is not None and _out_dim < y.shape[-1]) else None
        m = calc_metrics(y_target, y_hat, scaler=scaler_obj, channel_idx=_ch_idx)
        x_last = x[:, -1:, :, :_out_dim] if (_out_dim is not None and _out_dim < x.shape[-1]) else x[:, -1:, :, :]
        q = calc_quality_metrics(y_target, y_hat, x_last, scaler=scaler_obj, channel_idx=_ch_idx)

        bsz = x.size(0)
        seen += bsz
        meter["loss"]  += float(loss.item()) * bsz
        meter["pred"]  += float(parts["pred"].item()) * bsz
        meter["sparse"]+= float(parts["sparse"].item()) * bsz
        meter["var"]   += float(parts["var"].item()) * bsz
        meter["sym"]   += float(parts["sym"].item()) * bsz
        meter["vmg"]   += float(parts.get("vmg", torch.tensor(0.0)).item()) * bsz
        meter["qml"]   += float(parts.get("qml", torch.tensor(0.0)).item()) * bsz
        meter["delta"] += float(parts.get("delta", torch.tensor(0.0)).item()) * bsz
        meter["mae"]   += m["mae"] * bsz
        meter["rmse"]  += m["rmse"] * bsz
        meter["mape"] += m["mape"] * bsz
        meter["pred_std"] += q["pred_std"] * bsz
        meter["true_std"] += q["true_std"] * bsz
        meter["persistence_mae"] += q["persistence_mae"] * bsz
        meter["acc10"] += q["acc10"] * bsz
        meter["acc15"] += q["acc15"] * bsz

    for k in meter:
        meter[k] /= max(seen, 1)
    meter["diversity"] = meter["pred_std"] / max(meter["true_std"], 1e-8)
    meter["vs_persistence"] = meter["mae"] / max(meter["persistence_mae"], 1e-8)
    return meter


# ----------------------------
# train (with AMP + resume/finetune + checkpoints)
# ----------------------------
def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    A: torch.Tensor,
    SF: torch.Tensor,
    TF: torch.Tensor,
    n_nodes: int,
    in_dim: int,
    seq_len: int,
    pred_len: int,
    sf_dim: int,
    tf_dim: int,
    out_dim: int = None,
    d_model: int = 64,
    n_heads: int = 4,
    dropout: float = 0.1,
    ff_multiplier: int = 4,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 2,
    epochs: int = 50,                 # 训练“接下来”的轮数（若续训，就在start_epoch基础上再跑这么多）
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    scaler_obj=None,
    device: Optional[torch.device] = None,
    teacher_forcing: bool = False,
    best_ckpt_path: Optional[str] = None,
    last_ckpt_path: Optional[str] = None,   # 每轮覆盖保存
    # 断点续训 / 微调
    resume_ckpt: Optional[str] = None,      # 传已有 .pth 路径即可加载并继续
    resume_states: bool = False,            # True=同时恢复 optimizer/scheduler
    finetune_lr: Optional[float] = None,    # 微调时可重设新的较小 LR
    resume_best_mae: Optional[float] = None,# 恢复 best-ckpt 时显式播种历史最优 MAE
    # AMP / TF32
    use_amp: bool = True,
    amp_policy: str = "auto",               # 'auto' | 'bf16' | 'fp16' | 'off'
    allow_tf32: bool = True,                # 对 A100/30xx 打开 TF32 提速（不影响数值稳定）
    # 进度文件（独立于主 stdout log，便于监控脚本读取）
    progress_log: Optional[str] = None,     # 写入路径，None=不写
    log_interval: int = 5,                  # 每隔多少 epoch 写一行（best 改善时额外写）
    use_direct_pred: bool = False,          # True=直接预测头; False=自回归解码器
    # 空间混合模式
    spatial_mode: str = "attention",        # "attention" | "gcn" | "hybrid"
    k_hop: int = 8,                         # GCN/hybrid模式下的多跳邻接数
    # LR scheduler 选择
    scheduler_type: str = "cosine",         # "cosine" | "plateau"
    plateau_patience: int = 8,              # ReduceLROnPlateau patience
    eta_min: float = 1e-5,                  # cosine scheduler 最小学习率
    cosine_T0: int = 30,                    # CosineAnnealingWarmRestarts 首周期长度
    cosine_T_mult: int = 2,                 # 周期倍增因子: 1=固定周期, 2=递增(30→60→120)
    # 防坍缩
    diversity_weight: float = 0.0,          # 时间多样性损失权重 (>0 启用防坍缩)
    diversity_threshold: float = 0.3,       # pred_std/true_std 最低比例
    vmg_weight: float = 0.0,                # variance matching gradient (>0 启用，推荐 0.3)
    qml_weight: float = 0.0,                # quantile matching loss (IQR+tail, >0 启用，推荐 0.3)
    horizon_weight_mode: str = "uniform",   # uniform | linear | exp | final
    horizon_last_weight: float = 1.0,       # last horizon loss weight before mean normalization
    delta_weight: float = 0.0,              # trend/delta matching loss weight
    residual_delta: bool = True,            # False=不用残差修正, 强制模型预测绝对值(防半稳态数据坍缩)
    monitor_mode: str = "mae",              # "mae" | "strict"
    strict_diversity_gate: float = 0.5,     # strict monitor 的 diversity 门槛
    strict_vs_persistence_gate: float = 0.95,  # strict monitor 的 vs_persistence 门槛
    direct_head_mode: str = "legacy",       # "legacy" | "chunkwise" | "attn" | "linear"
    direct_step_refine: bool = False,       # 直接预测头的 chunk 内逐步残差细化
    decoder_future_tod: bool = True,        # 自回归解码时是否注入未来TOD；旧ckpt默认关闭
    gated_residual: bool = False,           # learn horizon/node residual gate instead of always adding last value
    ema_decay: float = 0.0,                 # >0 时启用参数EMA，用于收尾
    ema_start_epoch: int = 0,               # 从哪个epoch开始更新EMA
    ema_eval_interval: int = 1,             # 每隔多少epoch评估一次EMA
    grad_accum_steps: int = 1,              # batch 小时做梯度累积，提升有效 batch
    head_warmup_epochs: int = 0,            # warm-start 新预测头时先只训练 head
    head_lr_mult: float = 1.0,              # head 参数组学习率倍率
    roi_stop_epochs: int = 0,               # stop if recent MAE drop is too small
    roi_min_mae_drop: float = 0.0,          # minimum recent MAE drop over roi_stop_epochs
    allow_resume_mismatch: bool = False,    # True=允许大幅non-strict warm-start
    # 多卡并行
    gpu_ids: Optional[list] = None,         # e.g. [0,1] 表示用两张卡DataParallel
    scaler_type: Optional[str] = None,      # 保存到 checkpoint，供严格评估重建
    scale_target_only: bool = False,        # calendar channels stay raw while target channel is scaled
    calendar_features: str = "tod",         # saved audit metadata: tod or tod+dow
    split_rate: Optional[float] = None,     # 保存训练协议，供复现和评估审计
    val_ratio: Optional[float] = None,      # 保存训练协议，供复现和评估审计
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if out_dim is None:
        out_dim = in_dim
    use_future_time_features = bool(scale_target_only and in_dim > out_dim)
    if calendar_features == "tod+dow":
        calendar_feature_count = 2
    elif calendar_features == "tod":
        calendar_feature_count = 1
    else:
        calendar_feature_count = 0
    time_feature_start = (
        max(out_dim, in_dim - calendar_feature_count)
        if use_future_time_features and calendar_feature_count > 0
        else out_dim
    )

    # TF32 设置（只影响 matmul/conv 的内部精度，通常有益）
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)

    model = STFractalTransformer(
        n_nodes=n_nodes, in_dim=in_dim, d_model=d_model, n_heads=n_heads,
        seq_len=seq_len, pred_len=pred_len, sf_dim=sf_dim, tf_dim=tf_dim,
        dropout=dropout, ff_multiplier=ff_multiplier,
        num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
        use_bias_zscore=True, residual_delta=residual_delta,
        out_dim=out_dim,
        use_direct_pred=use_direct_pred,
        direct_head_mode=direct_head_mode,
        direct_step_refine=direct_step_refine,
        decoder_future_tod=decoder_future_tod,
        spatial_mode=spatial_mode,
        k_hop=k_hop,
        gated_residual=gated_residual,
        use_future_time_features=use_future_time_features,
        time_feature_start=time_feature_start,
    ).to(device)

    # torch.compile 已禁用（Transformer decoder的causal mask导致graph break，反而变慢）

    _print_model_profile(
        model,
        n_nodes=n_nodes, in_dim=in_dim, out_dim=out_dim, d_model=d_model, n_heads=n_heads,
        enc_layers=num_encoder_layers, dec_layers=num_decoder_layers,
        ff_mult=ff_multiplier, dropout=dropout,
        seq_len=seq_len, pred_len=pred_len,
        sf_dim=sf_dim, tf_dim=tf_dim
    )

    # ---- Multi-GPU DataParallel ----
    use_multigpu = gpu_ids is not None and len(gpu_ids) > 1 and device.type == "cuda"
    if use_multigpu:
        wrapper = _MultiGPUWrapper(model, A, SF, TF)
        model = nn.DataParallel(wrapper, device_ids=gpu_ids, output_device=gpu_ids[0])
        print(f"[MultiGPU] DataParallel on GPUs {gpu_ids}")

    # 全程使用小正则化权重（不使用warmup，避免训练中途突然激活导致性能退化）
    criterion = STFractalLoss(
        huber_delta=1.0, alpha=1.0, beta=0.01, phi=0.01, gamma=0.1, eta_sym=0.0,
        diversity_weight=diversity_weight, diversity_threshold=diversity_threshold,
        vmg_weight=vmg_weight, qml_weight=qml_weight,
        horizon_weight_mode=horizon_weight_mode,
        horizon_last_weight=horizon_last_weight,
        delta_weight=delta_weight,
    ).to(device)
    warmup_at_epoch = 99999  # 实际禁用warmup

    head_warmup_epochs = max(int(head_warmup_epochs), 0)
    grad_accum_steps = max(int(grad_accum_steps), 1)
    if head_warmup_epochs > 0:
        _set_head_bootstrap_trainable(model, enabled=True)
        trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_now = sum(p.numel() for p in model.parameters())
        print(
            f"[HeadBootstrap] enabled epochs={head_warmup_epochs} "
            f"trainable={trainable_now:,}/{total_now:,} head_lr_mult={head_lr_mult:g}"
        )
    if grad_accum_steps > 1:
        print(f"[GradAccum] steps={grad_accum_steps} effective_batch≈{grad_accum_steps}x loader_batch")

    # optimizer 需在 DataParallel 包装后创建，确保参数引用正确
    optimizer = _make_optimizer(model, lr=lr, weight_decay=weight_decay, head_lr_mult=head_lr_mult)
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cosine_T0, T_mult=cosine_T_mult, eta_min=eta_min
        )
        print(f"[Scheduler] CosineAnnealingWarmRestarts: T_0={cosine_T0}, T_mult={cosine_T_mult}, eta_min={eta_min}")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=plateau_patience
        )
        print(f"[Scheduler] ReduceLROnPlateau: factor=0.5, patience={plateau_patience}")

    # ---- AMP setup ----
    amp_enabled, amp_dtype, scaler = _resolve_amp(device, use_amp, amp_policy)
    print(f"[AMP] enabled={amp_enabled} dtype={amp_dtype} scaler={'on' if (scaler and scaler.is_enabled()) else 'off'}  "
          f"(TF32={'on' if (device.type=='cuda' and allow_tf32) else 'off'})")
    print(f"[StrictMetrics] enabled=1 progress_log={'on' if progress_log else 'off'} "
          f"fields=MAE,best_MAE,div,pers,vsp,acc10,acc15")
    print(f"[Monitor] mode={monitor_mode} strict_div={strict_diversity_gate} strict_vsp={strict_vs_persistence_gate} "
          f"direct_head_mode={direct_head_mode} direct_step_refine={int(bool(direct_step_refine))} "
          f"decoder_future_tod={int(bool(decoder_future_tod))} "
          f"gated_residual={int(bool(gated_residual))} "
          f"future_time={int(bool(use_future_time_features))} time_start={time_feature_start}")
    if ema_decay > 0.0:
        print(f"[EMA] enabled=1 decay={ema_decay:.6f} start_epoch={ema_start_epoch} eval_interval={ema_eval_interval}")
    if horizon_weight_mode != "uniform" or abs(float(horizon_last_weight) - 1.0) > 1e-8 or delta_weight > 0.0:
        print(
            f"[HorizonLoss] mode={horizon_weight_mode} last_weight={horizon_last_weight:g} "
            f"delta_weight={delta_weight:g}"
        )
    if roi_stop_epochs and roi_stop_epochs > 1:
        print(f"[ROIStop] window={roi_stop_epochs} min_mae_drop={roi_min_mae_drop:g}")

    # ---- resume / finetune ----
    start_epoch = 1
    best_val = float("inf")
    best_mae = float("inf")
    resumed_successfully = False
    if resume_ckpt is not None and os.path.isfile(resume_ckpt):
        resume_plan = resolve_resume_checkpoint_policy(
            resume_ckpt,
            resume_states=resume_states,
            resume_best_mae=resume_best_mae,
        )
        effective_resume_ckpt = resume_plan["path"] or resume_ckpt
        if effective_resume_ckpt != resume_ckpt:
            print(f"[Resume] Swap source checkpoint: {resume_ckpt} -> {effective_resume_ckpt} "
                  f"({resume_plan['reason']})")
        if resume_plan["source_best_mae"] is not None and resume_best_mae is None:
            print(f"[Resume] Infer source best_MAE={resume_plan['source_best_mae']:.4f} from progress log")

        print(f"[Resume] Loading checkpoint from: {effective_resume_ckpt}")
        _inner_for_resume = _get_inner_model(model)
        try:
            start_epoch, best_val_loaded, _cfg = load_checkpoint(
                effective_resume_ckpt, _inner_for_resume,
                optimizer if resume_states else None,
                scheduler if resume_states else None,
                map_location=device,
                strict=bool(resume_states),
                guard_model_only_mismatch=(not resume_states) and (not allow_resume_mismatch),
            )
            resumed_successfully = True
        except RuntimeError as exc:
            if resume_states or allow_resume_mismatch:
                raise
            print(f"[ResumeGuard] Disable resume and restart fresh: {exc}")
            resume_ckpt = None
            resume_plan = {"best_mae_seed": None, "source_best_mae": None, "reason": None}
            start_epoch = 1
            best_val_loaded = float("inf")
            _cfg = {}
        # 微调：不恢复优化器/调度器，且可重设更小的学习率
        if resumed_successfully and (not resume_states) and (finetune_lr is not None):
            for g in optimizer.param_groups:
                g["lr"] = finetune_lr * (head_lr_mult if g.get("name") == "head" else 1.0)
            print(f"[Finetune] Set optimizer LR to {finetune_lr:g}")
        if resumed_successfully:
            best_val, best_mae = resolve_resume_monitor_state(
                best_val_loaded,
                resume_states=resume_states,
                resume_best_mae=resume_plan["best_mae_seed"],
            )
            if (not resume_states) and np.isfinite(best_val_loaded):
                print(f"[Resume] Reset monitor baseline from loaded={best_val_loaded:.6f} because resume_mode=model")
            if resume_plan["best_mae_seed"] is not None:
                print(f"[Resume] Seed best_MAE to {float(resume_plan['best_mae_seed']):.4f}")
            print(f"[Resume] Start from epoch {start_epoch}, best_val={best_val:.6f}")
        if resumed_successfully and (not resume_states) and scheduler_type == "cosine":
            # Model-only resume should inherit the checkpoint's scheduler phase instead of
            # restarting warm restarts from max LR, which is especially harmful for plateaued
            # long-horizon finetuning branches.
            resume_sched_epoch = max(start_epoch - 1, 0)
            if resume_sched_epoch > 0:
                scheduler.step(resume_sched_epoch)
                print(
                    f"[Resume] Align cosine scheduler phase to epoch {resume_sched_epoch} "
                    f"(lr={optimizer.param_groups[0]['lr']:.2e})"
                )
    else:
        print("[Resume] No checkpoint. Train from scratch.")

    # 训练循环：从 start_epoch 开始，再训练 epochs 轮
    best_state = None
    wait = 0
    end_epoch = start_epoch + epochs - 1
    if resumed_successfully and progress_log and os.path.isfile(progress_log):
        try:
            with open(progress_log, "r") as _pf:
                for _line in reversed(_pf.readlines()):
                    if "best_MAE=" in _line:
                        for _tok in _line.strip().split():
                            if _tok.startswith("best_MAE="):
                                best_mae = min(best_mae, float(_tok.split("=", 1)[1]))
                                raise StopIteration
        except StopIteration:
            pass
        except Exception:
            pass
    ema_helper = None
    if ema_decay > 0.0:
        ema_helper = _ExponentialMovingAverage(_get_inner_model(model), decay=ema_decay)

    head_bootstrap_end_epoch = start_epoch + head_warmup_epochs - 1
    recent_effective_maes = []
    for epoch in range(start_epoch, end_epoch + 1):
        if head_warmup_epochs > 0 and epoch == head_bootstrap_end_epoch + 1:
            _set_head_bootstrap_trainable(model, enabled=False)
            print(f"[HeadBootstrap] epoch={epoch}: unfreeze backbone")

        # 到达/超过 warmup 点后一次性打开正则（若尚未打开）
        if epoch >= warmup_at_epoch and (criterion.beta == 0.0 or criterion.phi == 0.0):
            criterion.beta = 0.1
            criterion.phi = 0.1
            print(f"[Warmup->ON] epoch={epoch}: set beta={criterion.beta}, phi={criterion.phi}")

        t0 = time.time()
        tr = train_one_epoch(
            model, criterion, optimizer, train_loader, device, A, SF, TF, scaler_obj,
            grad_clip=1.0, teacher_forcing=teacher_forcing,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler,
            out_dim=out_dim,
            grad_accum_steps=grad_accum_steps,
        )
        va = evaluate(
            model, criterion, val_loader, device, A, SF, TF, scaler_obj,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype,
            out_dim=out_dim,
        )

        raw_monitor_value = compute_monitor_value(
            va,
            mode=monitor_mode,
            strict_diversity_gate=strict_diversity_gate,
            strict_vs_persistence_gate=strict_vs_persistence_gate,
        )
        effective_va = dict(va)
        effective_monitor_value = raw_monitor_value
        effective_source = "raw"
        ema_va = None
        _inner = _get_inner_model(model)
        if ema_helper is not None and epoch >= ema_start_epoch:
            ema_helper.update(_inner)
            if ((epoch - ema_start_epoch) % max(ema_eval_interval, 1)) == 0:
                raw_state = {k: v.detach().cpu() for k, v in _inner.state_dict().items()}
                _inner.load_state_dict(ema_helper.shadow, strict=True)
                ema_va = evaluate(
                    model, criterion, val_loader, device, A, SF, TF, scaler_obj,
                    amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                    out_dim=out_dim,
                )
                _inner.load_state_dict(raw_state, strict=True)
                ema_monitor_value = compute_monitor_value(
                    ema_va,
                    mode=monitor_mode,
                    strict_diversity_gate=strict_diversity_gate,
                    strict_vs_persistence_gate=strict_vs_persistence_gate,
                )
                if ema_monitor_value < effective_monitor_value - 1e-6:
                    effective_va = dict(ema_va)
                    effective_monitor_value = ema_monitor_value
                    effective_source = "ema"
        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(raw_monitor_value)
        t1 = time.time()

        cur_lr = optimizer.param_groups[0]["lr"]
        vmg_str = f" vmg={tr.get('vmg', 0):.4f}" if vmg_weight > 0 else ""
        qml_str = f" qml={tr.get('qml', 0):.4f}" if qml_weight > 0 else ""
        delta_str = f" delta={tr.get('delta', 0):.4f}" if delta_weight > 0 else ""
        print(f"[Epoch {epoch:03d}] "
              f"Train: loss={tr['loss']:.4f} pred={tr['pred']:.4f} sparse={tr['sparse']:.4f} var={tr['var']:.4f}{vmg_str}{qml_str}{delta_str} "
              f"| Val: loss={va['loss']:.4f} MAE={va['mae']:.4f} RMSE={va['rmse']:.4f} sMAPE={va['mape']:.4f} "
              f"Div={va['diversity']:.2f} vsP={va['vs_persistence']:.2f} Acc15={va['acc15']:.2%} "
              f"| lr={cur_lr:.2e} | time={t1-t0:.1f}s")
        if ema_va is not None:
            print(f"[EMA  {epoch:03d}] Val: loss={ema_va['loss']:.4f} MAE={ema_va['mae']:.4f} "
                  f"RMSE={ema_va['rmse']:.4f} Div={ema_va['diversity']:.2f} "
                  f"vsP={ema_va['vs_persistence']:.2f} source={effective_source}")

        # 写入独立进度文件（每 log_interval epoch 或 MAE 改善时）
        mae_improved = effective_va['mae'] < best_mae
        if mae_improved:
            best_mae = effective_va['mae']
        if progress_log and (epoch % log_interval == 0 or mae_improved):
            with open(progress_log, "a") as _pf:
                _pf.write(
                    f"epoch={epoch:03d} MAE={effective_va['mae']:.4f} best_MAE={best_mae:.4f} "
                    f"div={effective_va['diversity']:.4f} pers={effective_va['persistence_mae']:.4f} "
                    f"vsp={effective_va['vs_persistence']:.4f} acc10={effective_va['acc10']:.4f} "
                    f"acc15={effective_va['acc15']:.4f} source={effective_source} "
                    f"raw_MAE={va['mae']:.4f}"
                    + (f" ema_MAE={ema_va['mae']:.4f}" if ema_va is not None else "")
                    + "\n"
                )

        # 保存 last（每轮覆盖）— 统一用内部模型的 state_dict
        _ckpt_cfg = _pack_config(
            n_nodes=n_nodes, in_dim=in_dim, d_model=d_model, n_heads=n_heads,
            seq_len=seq_len, pred_len=pred_len, sf_dim=sf_dim, tf_dim=tf_dim,
            enc_layers=num_encoder_layers, dec_layers=num_decoder_layers,
            ff_mult=ff_multiplier, dropout=dropout,
            use_direct_pred=use_direct_pred, spatial_mode=spatial_mode, k_hop=k_hop,
            residual_delta=residual_delta, scaler=scaler_type,
            diversity_weight=diversity_weight, vmg_weight=vmg_weight, qml_weight=qml_weight,
            horizon_weight_mode=horizon_weight_mode,
            horizon_last_weight=horizon_last_weight,
            delta_weight=delta_weight,
            monitor_mode=monitor_mode, strict_diversity_gate=strict_diversity_gate,
            strict_vs_persistence_gate=strict_vs_persistence_gate,
            direct_head_mode=direct_head_mode,
            direct_step_refine=bool(direct_step_refine),
            decoder_future_tod=bool(decoder_future_tod),
            gated_residual=bool(gated_residual),
            use_future_time_features=bool(use_future_time_features),
            time_feature_start=int(time_feature_start),
            scale_target_only=bool(scale_target_only),
            calendar_features=calendar_features,
            split_rate=split_rate,
            val_ratio=val_ratio,
            grad_accum_steps=grad_accum_steps,
            head_warmup_epochs=head_warmup_epochs,
            head_lr_mult=head_lr_mult,
            roi_stop_epochs=roi_stop_epochs,
            roi_min_mae_drop=roi_min_mae_drop,
        )
        if last_ckpt_path:
            save_named_checkpoint(
                "last", last_ckpt_path, _inner,
                optimizer, scheduler,
                epoch=epoch, monitor_value=effective_monitor_value,
                config=_ckpt_cfg,
            )

        # 保存 best
        if effective_monitor_value < best_val - 1e-6:
            best_val = effective_monitor_value
            if effective_source == "ema" and ema_helper is not None:
                best_state = ema_helper.clone_state_dict()
            else:
                best_state = {k: v.detach().cpu() for k, v in _inner.state_dict().items()}
            wait = 0
            if best_ckpt_path:
                save_named_checkpoint(
                    "best", best_ckpt_path, _inner, None, None,
                    epoch=epoch, monitor_value=best_val,
                    config=_ckpt_cfg,
                    state_override=best_state,
                )
        else:
            wait += 1
            if roi_stop_epochs and roi_stop_epochs > 1:
                recent_effective_maes.append(float(effective_va["mae"]))
                if len(recent_effective_maes) > int(roi_stop_epochs):
                    recent_effective_maes.pop(0)
                if len(recent_effective_maes) == int(roi_stop_epochs):
                    recent_drop = recent_effective_maes[0] - min(recent_effective_maes)
                    if recent_drop < float(roi_min_mae_drop):
                        print(
                            f"[ROIStop] no meaningful MAE drop over {roi_stop_epochs} evals: "
                            f"drop={recent_drop:.6f} < {roi_min_mae_drop:.6f}"
                        )
                        break
            if wait >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        _get_inner_model(model).load_state_dict(best_state)

    return _get_inner_model(model)
