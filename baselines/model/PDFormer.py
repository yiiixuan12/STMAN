# -*- coding: utf-8 -*-
import math
import logging
from functools import partial
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Small utilities
# -----------------------------

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or (not training):
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob or 0.0

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# -----------------------------
# Basic Embeddings
# -----------------------------

class TokenEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        return self.norm(self.token_embed(x))


class PositionalEncoding(nn.Module):
    """
    动态生成正弦位置编码：不受 max_len 限制，自动适配 (T, D/设备/精度)。
    返回形状 (1, T, N, D) 以便与 (B, T, N, D) 广播相加。
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("pe", torch.zeros(1, 0, embed_dim), persistent=False)

    @torch.no_grad()
    def _build(self, T: int, device, dtype):
        position = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T,1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device, dtype=dtype)
            * (-(math.log(10000.0) / self.embed_dim))
        )  # (D/2,)
        pe = torch.zeros(T, self.embed_dim, device=device, dtype=dtype)  # (T,D)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1,T,D)

    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        if (
            self.pe.size(1) < T
            or self.pe.size(-1) != D
            or self.pe.device != x.device
            or self.pe.dtype != x.dtype
        ):
            self.pe = self._build(T, x.device, x.dtype)
        pe = self.pe[:, :T]                 # (1, T, D)
        pe = pe.unsqueeze(2).expand(1, T, N, D)  # (1, T, N, D)
        return pe


class LaplacianPE(nn.Module):
    """Linear projection of Laplacian eigenvectors (provided as lap_mx)."""
    def __init__(self, lape_dim: int, embed_dim: int):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx: torch.Tensor):
        # lap_mx: (N, lape_dim)
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx)  # (N, D)
        return lap_pos_enc.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)


class DataEmbedding(nn.Module):
    def __init__(
        self, feature_dim: int, embed_dim: int, lape_dim: int, adj_mx, drop: float = 0.0,
        add_time_in_day: bool = False, add_day_in_week: bool = False, device: torch.device = torch.device('cpu'),
    ):
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)

        self.position_encoding = PositionalEncoding(embed_dim)
        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, lap_mx: torch.Tensor):
        """
        x: (B, T, N, C) with optional extra channels for time/day one-hots
        lap_mx: (N, lape_dim)
        """
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x = x + self.position_encoding(x)
        if self.add_time_in_day:
            tid = (origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long()
            x = x + self.daytime_embedding(tid.clamp_(0, self.minute_size - 1))
        if self.add_day_in_week:
            week_oh = origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8]
            weekday_idx = week_oh.argmax(dim=3)
            x = x + self.weekday_embedding(weekday_idx)
        x = x + self.spatial_embedding(lap_mx.to(x.device))
        return self.dropout(x)


# -----------------------------
# Attention blocks
# -----------------------------

class STSelfAttention(nn.Module):
    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1,
    ):
        super().__init__()
        total_heads = geo_num_heads + sem_num_heads + t_num_heads
        assert total_heads > 0 and dim % total_heads == 0, \
            f"dim={dim} must be divisible by enabled heads ({total_heads})."

        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.t_num_heads   = t_num_heads

        self.head_dim = dim // total_heads
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.s_attn_size = s_attn_size
        self.t_attn_size = t_attn_size
        self.output_dim = output_dim

        geo_ch = self.head_dim * self.geo_num_heads
        sem_ch = self.head_dim * self.sem_num_heads
        t_ch   = self.head_dim * self.t_num_heads

        self.has_geo = geo_ch > 0
        self.has_sem = sem_ch > 0
        self.has_time = t_ch > 0

        if self.has_geo:
            self.pattern_q_linears = nn.ModuleList([nn.Linear(dim, geo_ch) for _ in range(output_dim)])
            self.pattern_k_linears = nn.ModuleList([nn.Linear(dim, geo_ch) for _ in range(output_dim)])
            self.pattern_v_linears = nn.ModuleList([nn.Linear(dim, geo_ch) for _ in range(output_dim)])

            self.geo_q_conv = nn.Conv2d(dim, geo_ch, kernel_size=1, bias=qkv_bias)
            self.geo_k_conv = nn.Conv2d(dim, geo_ch, kernel_size=1, bias=qkv_bias)
            self.geo_v_conv = nn.Conv2d(dim, geo_ch, kernel_size=1, bias=qkv_bias)
            self.geo_attn_drop = nn.Dropout(attn_drop)

        if self.has_sem:
            self.sem_q_conv = nn.Conv2d(dim, sem_ch, kernel_size=1, bias=qkv_bias)
            self.sem_k_conv = nn.Conv2d(dim, sem_ch, kernel_size=1, bias=qkv_bias)
            self.sem_v_conv = nn.Conv2d(dim, sem_ch, kernel_size=1, bias=qkv_bias)
            self.sem_attn_drop = nn.Dropout(attn_drop)

        if self.has_time:
            self.t_q_conv = nn.Conv2d(dim, t_ch, kernel_size=1, bias=qkv_bias)
            self.t_k_conv = nn.Conv2d(dim, t_ch, kernel_size=1, bias=qkv_bias)
            self.t_v_conv = nn.Conv2d(dim, t_ch, kernel_size=1, bias=qkv_bias)
            self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        B, T, N, D = x.shape
        outs = []

        # ---- temporal ----
        if self.has_time:
            t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
            t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
            t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
            t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
            t_attn = t_attn.softmax(dim=-1)
            t_attn = self.t_attn_drop(t_attn)
            t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, self.head_dim * self.t_num_heads).transpose(1, 2)
            outs.append(t_x)

        # ---- geo + pattern ----
        if self.has_geo:
            geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            for i in range(self.output_dim):
                pattern_q = self.pattern_q_linears[i](x_patterns[..., i])
                pattern_k = self.pattern_k_linears[i](pattern_keys[..., i])
                pattern_v = self.pattern_v_linears[i](pattern_keys[..., i])
                pattern_attn = (pattern_q @ pattern_k.transpose(-2, -1)) * self.scale
                pattern_attn = pattern_attn.softmax(dim=-1)
                geo_k = geo_k + pattern_attn @ pattern_v
            geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            geo_q = geo_q.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            geo_k = geo_k.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            geo_v = geo_v.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

            geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale
            if geo_mask is not None:
                geo_attn = geo_attn.masked_fill(geo_mask[None, None, None, :, :], float('-inf'))
            geo_attn = geo_attn.softmax(dim=-1)
            geo_attn = self.geo_attn_drop(geo_attn)
            geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N, self.head_dim * self.geo_num_heads)
            outs.append(geo_x)

        # ---- semantic ----
        if self.has_sem:
            sem_q = self.sem_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            sem_k = self.sem_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            sem_v = self.sem_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            sem_q = sem_q.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            sem_k = sem_k.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            sem_v = sem_v.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
            if sem_mask is not None:
                sem_attn = sem_attn.masked_fill(sem_mask[None, None, None, :, :], float('-inf'))
            sem_attn = sem_attn.softmax(dim=-1)
            sem_attn = self.sem_attn_drop(sem_attn)
            sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, N, self.head_dim * self.sem_num_heads)
            outs.append(sem_x)

        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        out = self.proj(out)
        return self.proj_drop(out)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class STEncoderBlock(nn.Module):
    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'), type_ln="pre", output_dim=1,
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.st_attn = STSelfAttention(
            dim, s_attn_size, t_attn_size, geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, t_num_heads=t_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, device=device, output_dim=output_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        if self.type_ln == 'pre':
            x = x + self.drop_path(self.st_attn(self.norm1(x), x_patterns, pattern_keys, geo_mask=geo_mask, sem_mask=sem_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:  # 'post'
            x = self.norm1(x + self.drop_path(self.st_attn(x, x_patterns, pattern_keys, geo_mask=geo_mask, sem_mask=sem_mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


# -----------------------------
# Minimal scaler & loss utils
# -----------------------------

class IdentityScaler:
    def inverse_transform(self, x):
        return x


def _masked(y, null_val=None):
    if null_val is None:
        mask = ~torch.isnan(y)
    else:
        mask = (y != null_val) & ~torch.isnan(y)
    mask = mask.float()
    mask = mask / (mask.mean() + 1e-12)
    return mask


def masked_mae_torch(y_pred, y_true, null_val=None):
    mask = _masked(y_true, null_val)
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    return torch.nan_to_num(loss).mean()


def masked_mse_torch(y_pred, y_true, null_val=None):
    mask = _masked(y_true, null_val)
    loss = (y_pred - y_true) ** 2
    loss = loss * mask
    return torch.nan_to_num(loss).mean()


def masked_rmse_torch(y_pred, y_true, null_val=None):
    return torch.sqrt(masked_mse_torch(y_pred, y_true, null_val) + 1e-12)


def masked_mape_torch(y_pred, y_true, null_val=None, eps: float = 1e-5):
    mask = _masked(y_true, null_val)
    loss = torch.abs((y_pred - y_true) / (y_true.abs() + eps))
    loss = loss * mask
    return torch.nan_to_num(loss).mean()


def log_cosh_loss(y_pred, y_true):
    x = y_pred - y_true
    return torch.mean(torch.log(torch.cosh(x + 1e-12)))


def huber_loss(y_pred, y_true, delta: float = 1.0):
    x = torch.abs(y_pred - y_true)
    quadratic = torch.minimum(x, torch.tensor(delta, device=x.device))
    linear = x - quadratic
    return torch.mean(0.5 * quadratic ** 2 + delta * linear)


def quantile_loss(y_pred, y_true, delta: float = 0.25):
    e = y_true - y_pred
    return torch.mean(torch.maximum(delta * e, (delta - 1) * e))


def masked_huber_loss(y_pred, y_true, delta: float = 1.0, null_val=None):
    mask = _masked(y_true, null_val)
    x = torch.abs(y_pred - y_true)
    quadratic = torch.minimum(x, torch.tensor(delta, device=x.device))
    linear = x - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return torch.nan_to_num(loss * mask).mean()


def r2_score_torch(y_pred, y_true, eps: float = 1e-12):
    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / (ss_tot + eps)


def explained_variance_score_torch(y_pred, y_true, eps: float = 1e-12):
    var_y = torch.var(y_true)
    var_err = torch.var(y_true - y_pred)
    return 1 - var_err / (var_y + eps)


# -----------------------------
# PDFormer (standalone)
# -----------------------------

class PDFormer(nn.Module):
    """
    Standalone PDFormer without libcity dependencies.
    Keep the original constructor signature (config, data_feature) for drop-in.
    """
    def __init__(self, config: Dict[str, Any], data_feature: Dict[str, Any]):
        super().__init__()

        self._logger = logging.getLogger(__name__)
        self.data_feature = data_feature
        self.num_nodes = data_feature.get("num_nodes", 1)
        self.feature_dim = data_feature.get("feature_dim", 1)
        self.ext_dim = data_feature.get("ext_dim", 0)
        self.num_batches = data_feature.get('num_batches', 1)
        self.dtw_matrix = data_feature.get('dtw_matrix')
        self.adj_mx = data_feature.get('adj_mx')
        sd_mx = data_feature.get('sd_mx')
        sh_mx = data_feature.get('sh_mx')

        self._scaler = data_feature.get('scaler', IdentityScaler())

        self.embed_dim = config.get('embed_dim', 64)
        self.skip_dim = config.get("skip_dim", 256)
        lape_dim = config.get('lape_dim', 8)
        geo_num_heads = config.get('geo_num_heads', 4)
        sem_num_heads = config.get('sem_num_heads', 2)
        t_num_heads = config.get('t_num_heads', 2)
        mlp_ratio = config.get("mlp_ratio", 4.0)
        qkv_bias = config.get("qkv_bias", True)
        drop = config.get("drop", 0.0)
        attn_drop = config.get("attn_drop", 0.0)
        drop_path = config.get("drop_path", 0.3)
        self.s_attn_size = config.get("s_attn_size", 3)
        self.t_attn_size = config.get("t_attn_size", 3)
        enc_depth = config.get("enc_depth", 6)
        type_ln = config.get("type_ln", "pre")
        self.type_short_path = config.get("type_short_path", "hop")

        self.output_dim = config.get('output_dim', 1)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)
        add_time_in_day = config.get("add_time_in_day", True)
        add_day_in_week = config.get("add_day_in_week", True)
        self.device_t = config.get('device', torch.device('cpu'))
        self.world_size = config.get('world_size', 1)
        self.huber_delta = config.get('huber_delta', 1.0)
        self.quan_delta = config.get('quan_delta', 0.25)
        self.far_mask_delta = config.get('far_mask_delta', 5.0)
        self.dtw_delta = config.get('dtw_delta', 5)

        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        self.step_size = config.get('step_size', 2500)
        self.max_epoch = config.get('max_epoch', 200)
        self.task_level = config.get('task_level', 0)

        if self.max_epoch * self.num_batches * self.world_size < self.step_size * self.output_window:
            self._logger.warning('Parameter `step_size` may be too big with {} epochs; '
                                 'the model may not be trained for all time steps.'.format(self.max_epoch))

        # build masks
        if self.type_short_path == "dist":
            distances = sd_mx[~np.isinf(sd_mx)].flatten()
            std = distances.std() if distances.size > 0 else 1.0
            sd_mx = np.exp(-np.square(sd_mx / (std + 1e-12)))
            far_mask = torch.zeros(self.num_nodes, self.num_nodes)
            far_mask[torch.from_numpy(sd_mx) < self.far_mask_delta] = 1
            self.geo_mask = far_mask.bool().to(self.device_t)
            self.sem_mask = torch.zeros_like(self.geo_mask, dtype=torch.bool)
        else:
            sh_mx = sh_mx.T
            self.geo_mask = torch.zeros(self.num_nodes, self.num_nodes)
            self.geo_mask[torch.from_numpy(sh_mx) >= self.far_mask_delta] = 1
            self.geo_mask = self.geo_mask.bool().to(self.device_t)

            self.sem_mask = torch.ones(self.num_nodes, self.num_nodes)
            sem_mask_np = self.dtw_matrix.argsort(axis=1)[:, :self.dtw_delta]
            for i in range(self.sem_mask.shape[0]):
                self.sem_mask[i][torch.from_numpy(sem_mask_np[i])] = 0
            self.sem_mask = self.sem_mask.bool().to(self.device_t)

        # pattern keys
        self.pattern_keys = torch.from_numpy(data_feature.get('pattern_keys')).float().to(self.device_t)
        self.pattern_embeddings = nn.ModuleList([
            TokenEmbedding(self.s_attn_size, self.embed_dim) for _ in range(self.output_dim)
        ])

        self.enc_embed_layer = DataEmbedding(
            self.feature_dim - self.ext_dim, self.embed_dim, lape_dim, self.adj_mx, drop=drop,
            add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week, device=self.device_t,
        )

        # encoder
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, s_attn_size=self.s_attn_size, t_attn_size=self.t_attn_size,
                geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, t_num_heads=t_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=enc_dpr[i],
                act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                device=self.device_t, type_ln=type_ln, output_dim=self.output_dim,
            ) for i in range(enc_depth)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1)
            for _ in range(enc_depth)
        ])

        # 时间维映射：T_in -> r -> T_out（比直接 T_in->T_out 更省参/快）
        r = min(64, self.input_window, self.output_window)  # 也可设成超参
        self.end_t_down = nn.Conv2d(self.input_window, r, kernel_size=1, bias=False)
        self.end_t_up   = nn.Conv2d(r, self.output_window, kernel_size=1, bias=True)

        self.end_conv2 = nn.Conv2d(
            in_channels=self.skip_dim, out_channels=self.output_dim, kernel_size=1, bias=True,
        )

    # ---------- forward & predict ----------
    def forward(self, batch: Dict[str, torch.Tensor], lap_mx: Optional[torch.Tensor] = None):
        """
        batch['X']: (B, Tin, N, C)
        Returns: (B, Tout, N, output_dim)
        """
        x = batch['X']
        B, T, N, C = x.shape
        assert N == self.num_nodes, f"num_nodes mismatch: {N} vs {self.num_nodes}"
        T_in = T

        # build pattern tensors
        x_pattern_list = []
        for i in range(self.s_attn_size):
            x_pattern = F.pad(
                x[:, :T_in + i + 1 - self.s_attn_size, :, :self.output_dim],
                (0, 0, 0, 0, self.s_attn_size - 1 - i, 0),
                "constant", 0,
            ).unsqueeze(-2)
            x_pattern_list.append(x_pattern)
        x_patterns = torch.cat(x_pattern_list, dim=-2)  # (B, T, N, s_attn_size, output_dim)

        x_pattern_list2 = []
        pattern_key_list = []
        for i in range(self.output_dim):
            x_pattern_list2.append(self.pattern_embeddings[i](x_patterns[..., i]).unsqueeze(-1))
            pattern_key_list.append(self.pattern_embeddings[i](self.pattern_keys[..., i]).unsqueeze(-1))
        x_patterns = torch.cat(x_pattern_list2, dim=-1)
        pattern_keys = torch.cat(pattern_key_list, dim=-1)

        # encoder
        if lap_mx is None:
            raise ValueError("lap_mx (N, lape_dim) is required for Laplacian positional encoding.")
        enc = self.enc_embed_layer(x, lap_mx)
        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            enc = encoder_block(enc, x_patterns, pattern_keys, self.geo_mask, self.sem_mask)
            skip = skip + self.skip_convs[i](enc.permute(0, 3, 2, 1))  # (B, skip_dim, N, T)

        # 时间维 1x1 卷积：先把 T 放到通道维 (B, T, N, skip_dim)
        skip = F.relu(skip.permute(0, 3, 2, 1))                 # (B, T_in, N, skip_dim)
        skip = self.end_t_up(F.relu(self.end_t_down(skip)))     # (B, T_out, N, skip_dim)

        # 输出维度映射到 out_dim
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))  # (B, out_dim, N, T_out)
        return skip.permute(0, 3, 2, 1)                          # (B, T_out, N, out_dim)

    def predict(self, batch: Dict[str, torch.Tensor], lap_mx: Optional[torch.Tensor] = None):
        return self.forward(batch, lap_mx)

    # ---------- loss selection (no libcity) ----------
    def get_loss_func(self, set_loss: str):
        s = (set_loss or 'masked_mae').lower()
        if s == 'mae':
            lf = masked_mae_torch
        elif s == 'mse':
            lf = masked_mse_torch
        elif s == 'rmse':
            lf = masked_rmse_torch
        elif s == 'mape':
            lf = masked_mape_torch
        elif s == 'logcosh':
            lf = log_cosh_loss
        elif s == 'huber':
            lf = partial(huber_loss, delta=self.huber_delta)
        elif s == 'quantile':
            lf = partial(quantile_loss, delta=self.quan_delta)
        elif s == 'masked_mae':
            lf = partial(masked_mae_torch, null_val=0)
        elif s == 'masked_mse':
            lf = partial(masked_mse_torch, null_val=0)
        elif s == 'masked_rmse':
            lf = partial(masked_rmse_torch, null_val=0)
        elif s == 'masked_mape':
            lf = partial(masked_mape_torch, null_val=0)
        elif s == 'masked_huber':
            lf = partial(masked_huber_loss, delta=self.huber_delta, null_val=0)
        elif s == 'r2':
            lf = r2_score_torch
        elif s == 'evar':
            lf = explained_variance_score_torch
        else:
            self._logger.warning(f'Unrecognized loss "{set_loss}", default to masked_mae.')
            lf = partial(masked_mae_torch, null_val=0)
        return lf

    def calculate_loss_without_predict(self, y_true: torch.Tensor, y_predicted: torch.Tensor,
                                       batches_seen: Optional[int] = None, set_loss: str = 'masked_mae'):
        lf = self.get_loss_func(set_loss=set_loss)
        try:
            y_true_inv = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_pred_inv = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        except Exception:
            y_true_inv, y_pred_inv = y_true[..., :self.output_dim], y_predicted[..., :self.output_dim]

        if self.training:
            if batches_seen is not None and self.step_size > 0:
                if (batches_seen % self.step_size == 0) and (self.task_level < self.output_window):
                    self.task_level += 1
                    self._logger.info('Training: task_level -> {}'.format(self.task_level))
            if self.use_curriculum_learning and self.task_level > 0:
                return lf(y_pred_inv[:, :self.task_level, :, :], y_true_inv[:, :self.task_level, :, :])
            else:
                return lf(y_pred_inv, y_true_inv)
        else:
            return lf(y_pred_inv, y_true_inv)

    def calculate_loss(self, batch: Dict[str, torch.Tensor], batches_seen: Optional[int] = None,
                       lap_mx: Optional[torch.Tensor] = None, set_loss: str = 'masked_mae'):
        y_true = batch['y']
        y_pred = self.predict(batch, lap_mx)
        return self.calculate_loss_without_predict(y_true, y_pred, batches_seen, set_loss=set_loss)
