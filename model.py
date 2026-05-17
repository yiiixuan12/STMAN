# model.py
# -*- coding: utf-8 -*-
import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# PositionalEncoding (time-only; support odd d_model)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, T, N, D = x.shape
            return x + self.pe[:T, :].view(1, T, 1, D)
        elif x.dim() == 3:
            B, T, D = x.shape
            return x + self.pe[:T, :].view(1, T, D)
        else:
            raise ValueError("PositionalEncoding expects 3D or 4D input")


# ----------------------------
# Utilities
# ----------------------------
def pairwise_cosine(e: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    e = F.normalize(e, p=2, dim=-1, eps=eps)
    return e @ e.T


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean_x = x.mean()
    std_x = x.std(unbiased=False)
    if std_x < eps:
        return torch.zeros_like(x)
    return (x - mean_x) / (std_x + eps)


# ----------------------------
# Spatial Graph Convolution with Fractal Bias
# ----------------------------
class SpatialMixing(nn.Module):
    """
    空间混合层：利用分形偏置引导的图卷积实现节点间信息交换。
    在每个时间步独立地进行空间聚合。
    S = softmax(B)   B是分形注意力偏置矩阵[N,N]
    X' = LayerNorm(X + W_o * (S @ X @ W_v))
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, N, d]
        S: [N, N]  softmax后的空间注意力权重
        return: [B, T, N, d]
        """
        v = self.W_v(x)                           # [B, T, N, d]
        agg = torch.einsum('mn,btnd->btmd', S, v)  # [B, T, N, d]  空间聚合
        out = self.W_o(self.dropout(agg))          # [B, T, N, d]
        return self.ln(x + out)                    # 残差 + LayerNorm


class SpatialMixingGCN(nn.Module):
    """
    多跳图卷积空间混合层，专为稀疏图设计（如PEMS07链式结构）。
    使用K-hop邻接多项式滤波器替代密集softmax注意力。
    支持可学习的多跳权重，让模型自适应确定每一跳的重要性。
    X' = LayerNorm(X + W_o * (A_khop @ X @ W_v))
    """
    def __init__(self, d_model: int, k_hop: int = 8, dropout: float = 0.1):
        super().__init__()
        self.k_hop = k_hop
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # 可学习的每跳权重（softmax归一化）
        self.hop_weights = nn.Parameter(torch.zeros(k_hop + 1))

    def forward(self, x: torch.Tensor, A_hops: list) -> torch.Tensor:
        """
        x: [B, T, N, d]
        A_hops: list of [N, N] tensors, A_hops[k] = normalized A^k (k=0..K)
        return: [B, T, N, d]
        """
        v = self.W_v(x)  # [B, T, N, d]
        # 对每一跳的聚合结果进行加权求和
        w = F.softmax(self.hop_weights, dim=0)
        agg = torch.zeros_like(v)
        for k, A_k in enumerate(A_hops):
            agg = agg + w[k] * torch.einsum('mn,btnd->btmd', A_k, v)
        out = self.W_o(self.dropout(agg))
        return self.ln(x + out)


class SpatialMixingHybrid(nn.Module):
    """
    Hybrid spatial mixer for large sparse traffic graphs.
    It keeps local multi-hop graph propagation while retaining the learned
    fractal attention matrix as a long-range correction.
    """
    def __init__(self, d_model: int, k_hop: int = 8, dropout: float = 0.1):
        super().__init__()
        self.k_hop = k_hop
        self.W_v_gcn = nn.Linear(d_model, d_model, bias=False)
        self.W_o_gcn = nn.Linear(d_model, d_model, bias=False)
        self.W_v_attn = nn.Linear(d_model, d_model, bias=False)
        self.W_o_attn = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.hop_weights = nn.Parameter(torch.zeros(k_hop + 1))
        self.gcn_mix_logit = nn.Parameter(torch.tensor(0.4))

    def forward(self, x: torch.Tensor, supports) -> torch.Tensor:
        """
        x: [B, T, N, d]
        supports: (S_attention [N, N], A_hops list[[N, N]])
        return: [B, T, N, d]
        """
        S_attn, A_hops = supports

        v_gcn = self.W_v_gcn(x)
        w = F.softmax(self.hop_weights, dim=0)
        gcn_agg = torch.zeros_like(v_gcn)
        for k, A_k in enumerate(A_hops):
            gcn_agg = gcn_agg + w[k] * torch.einsum('mn,btnd->btmd', A_k, v_gcn)
        gcn_update = self.W_o_gcn(self.dropout(gcn_agg))

        v_attn = self.W_v_attn(x)
        attn_agg = torch.einsum('mn,btnd->btmd', S_attn, v_attn)
        attn_update = self.W_o_attn(self.dropout(attn_agg))

        alpha = torch.sigmoid(self.gcn_mix_logit)
        out = alpha * gcn_update + (1.0 - alpha) * attn_update
        return self.ln(x + out)


# ----------------------------
# Spatiotemporal Encoder Block
# ----------------------------
class STEncoderBlock(nn.Module):
    """
    时空编码器块：先做时间自注意力，再做空间图卷积。
    支持三种空间混合模式: attention(密集注意力), gcn(多跳图卷积),
    hybrid(多跳图卷积+分形注意力融合)。
    """
    def __init__(self, d_model: int, n_heads: int, ff_multiplier: int = 4,
                 dropout: float = 0.1, spatial_mode: str = "attention",
                 k_hop: int = 8):
        super().__init__()
        self.spatial_mode = spatial_mode
        # 时间自注意力
        self.temporal_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff_multiplier * d_model,
            dropout=dropout, batch_first=False,
            activation="gelu", norm_first=True,
        )
        # 空间混合: 根据模式选择
        if spatial_mode == "gcn":
            self.spatial_mix = SpatialMixingGCN(d_model, k_hop=k_hop, dropout=dropout)
        elif spatial_mode == "hybrid":
            self.spatial_mix = SpatialMixingHybrid(d_model, k_hop=k_hop, dropout=dropout)
        else:
            self.spatial_mix = SpatialMixing(d_model, dropout)

    def forward(self, x_4d: torch.Tensor, S_spatial) -> torch.Tensor:
        """
        x_4d: [B, T, N, d]
        S_spatial: [N, N] for attention mode, list of [N, N] for gcn mode,
                   or (attention, hops) for hybrid mode
        return: [B, T, N, d]
        """
        B, T, N, d = x_4d.shape

        # 1) 时间自注意力: reshape to [T, B*N, d]
        x_t = x_4d.permute(1, 0, 2, 3).contiguous().view(T, B * N, d)
        x_t = self.temporal_attn(x_t)  # [T, B*N, d]
        x_4d = x_t.view(T, B, N, d).permute(1, 0, 2, 3).contiguous()  # [B, T, N, d]

        # 2) 空间混合
        x_4d = self.spatial_mix(x_4d, S_spatial)  # [B, T, N, d]

        return x_4d


# ----------------------------
# Bidirectional Spatio-Temporal Attention (node-level, multi-head)
# ----------------------------
class STBidirectionalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1,
                 use_bias_zscore: bool = True, bias_eps: float = 1e-3):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_bias_zscore = use_bias_zscore
        self.bias_eps = bias_eps
        self.dropout = nn.Dropout(dropout)

        self.W_Q_s = nn.Linear(d_model, d_model, bias=False)
        self.W_K_t = nn.Linear(d_model, d_model, bias=False)
        self.W_V_t = nn.Linear(d_model, d_model, bias=False)
        self.W_O_st = nn.Linear(d_model, d_model, bias=False)

        self.W_Q_t = nn.Linear(d_model, d_model, bias=False)
        self.W_K_s = nn.Linear(d_model, d_model, bias=False)
        self.W_V_s = nn.Linear(d_model, d_model, bias=False)
        self.W_O_ts = nn.Linear(d_model, d_model, bias=False)

        self._lambda_st = nn.Parameter(torch.zeros(3))
        self._lambda_ts = nn.Parameter(torch.zeros(3))

    @staticmethod
    def _split_heads(x, n_heads):
        N, D = x.shape
        return x.view(N, n_heads, D // n_heads).permute(1, 0, 2).contiguous()

    @staticmethod
    def _combine_heads(x):
        return x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)

    def _build_bias(self, A, E_s, E_t, lambdas):
        A = A.to(dtype=E_s.dtype)
        adj_bias = torch.log(A.clamp_min(self.bias_eps))
        cos_s = pairwise_cosine(E_s)
        cos_t = pairwise_cosine(E_t)
        if self.use_bias_zscore:
            adj_bias = zscore(adj_bias)
            cos_s = zscore(cos_s)
            cos_t = zscore(cos_t)
        lam = F.softplus(lambdas)
        return lam[0] * adj_bias + lam[1] * cos_s + lam[2] * cos_t

    def _attend(self, Q, K, V, B):
        h, d_k = self.n_heads, self.d_k
        q = self._split_heads(Q, h)
        k = self._split_heads(K, h)
        v = self._split_heads(V, h)
        logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d_k)
        logits = logits + B.unsqueeze(0)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = self._combine_heads(torch.matmul(attn, v))
        return out, attn.mean(dim=0)

    def forward(self, E_s, E_t, A):
        B_st = self._build_bias(A, E_s, E_t, self._lambda_st)
        B_ts = self._build_bias(A, E_s, E_t, self._lambda_ts)

        H_st, S_s2t = self._attend(self.W_Q_s(E_s), self.W_K_t(E_t), self.W_V_t(E_t), B_st)
        H_st = self.W_O_st(H_st)

        H_ts, S_t2s = self._attend(self.W_Q_t(E_t), self.W_K_s(E_s), self.W_V_s(E_s), B_ts)
        H_ts = self.W_O_ts(H_ts)

        return torch.cat([H_st, H_ts], dim=-1), S_s2t, S_t2s, B_st, B_ts


# ----------------------------
# Main Model
# ----------------------------
class STFractalTransformer(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        in_dim: int,
        d_model: int,
        n_heads: int,
        seq_len: int,
        pred_len: int,
        sf_dim: int,
        tf_dim: int,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        use_bias_zscore: bool = True,
        residual_delta: bool = True,
        out_dim: int = None,
        use_direct_pred: bool = False,
        direct_head_mode: str = "legacy",
        direct_step_refine: bool = False,
        decoder_future_tod: bool = True,
        spatial_mode: str = "attention",  # "attention" | "gcn" | "hybrid"
        k_hop: int = 8,                   # GCN模式下的多跳数
        gated_residual: bool = False,
        use_future_time_features: bool = False,
        time_feature_start: int = None,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.residual_delta = residual_delta
        self.use_direct_pred = use_direct_pred
        self.direct_head_mode = direct_head_mode
        self.direct_step_refine = bool(direct_step_refine and use_direct_pred and direct_head_mode != "linear")
        self.decoder_future_tod = bool(decoder_future_tod)
        self.spatial_mode = spatial_mode
        self.k_hop = k_hop
        self.gated_residual = bool(gated_residual and residual_delta)
        self.time_feature_start = int(time_feature_start) if time_feature_start is not None else self.out_dim
        self.time_feature_start = max(0, min(self.time_feature_start, in_dim))
        self.time_feature_dim = max(0, in_dim - self.time_feature_start)
        self.use_future_time_features = bool(use_future_time_features and self.time_feature_dim > 0)
        self.tod_idx = self.time_feature_start if self.time_feature_dim > 0 else None

        # 1) 分形嵌入
        self.sf_proj = nn.Sequential(nn.Linear(sf_dim, d_model), nn.Dropout(dropout))
        self.tf_proj = nn.Sequential(nn.Linear(tf_dim, d_model), nn.Dropout(dropout))
        self.ln_sf = nn.LayerNorm(d_model)
        self.ln_tf = nn.LayerNorm(d_model)

        # 2) 双向注意力
        self.biattn = STBidirectionalAttention(
            d_model=d_model, n_heads=n_heads,
            dropout=dropout, use_bias_zscore=use_bias_zscore, bias_eps=1e-3
        )

        # 3) 融合
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.LayerNorm(d_model),
        )

        # 4) 投影
        self.in_proj = nn.Linear(in_dim, d_model)
        self.out_hidden = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, self.out_dim)

        # 5) 位置编码（编码器共用）
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max(seq_len, pred_len) + 512)

        # 6) 时空编码器：每层包含 时间自注意力 + 空间混合（attention或gcn）
        self.st_encoder = nn.ModuleList([
            STEncoderBlock(d_model, n_heads, ff_multiplier, dropout,
                           spatial_mode=spatial_mode, k_hop=k_hop)
            for _ in range(num_encoder_layers)
        ])

        # 7) 解码器 或 直接预测头（二选一）
        if use_direct_pred:
            # 分组预测：将 pred_len 分成 n_chunks，每 chunk 用独立 embedding + 共享 MLP
            self._dp_chunk_size = min(12, pred_len)  # 每组最多12步
            self._dp_n_chunks = (pred_len + self._dp_chunk_size - 1) // self._dp_chunk_size
            self._dp_last_chunk = pred_len - self._dp_chunk_size * (self._dp_n_chunks - 1)
            if direct_head_mode == "chunkwise":
                # 为每个未来 chunk 学一组独立的时间聚合权重，避免把整段历史压成单点
                self.temporal_chunk_proj = nn.Linear(seq_len, self._dp_n_chunks, bias=False)
                self.chunk_emb = nn.Embedding(self._dp_n_chunks, d_model)
                self.future_time_proj = nn.Linear(self.time_feature_dim, d_model, bias=False) if self.use_future_time_features else None
                self.chunk_mlp = nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, self._dp_chunk_size * self.out_dim),
                )
            elif direct_head_mode == "attn":
                # 为每个未来 chunk 学一个查询向量，对编码历史做内容自适应聚合。
                # 相比固定 temporal_chunk_proj，它能在平台期任务中重新选择有效历史片段。
                self.chunk_query = nn.Embedding(self._dp_n_chunks, d_model)
                self.temporal_key = nn.Linear(d_model, d_model, bias=False)
                self.chunk_emb = nn.Embedding(self._dp_n_chunks, d_model)
                self.future_time_proj = nn.Linear(self.time_feature_dim, d_model, bias=False) if self.use_future_time_features else None
                self.chunk_mlp = nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, self._dp_chunk_size * self.out_dim),
                )
            elif direct_head_mode == "linear":
                # 兼容早期 v11/v6 系列 checkpoint：单 temporal_agg + 线性直接输出全 horizon
                self.temporal_agg = nn.Linear(seq_len, 1, bias=False)
                self.out_proj_direct = nn.Linear(d_model, pred_len * self.out_dim)
            else:
                # 兼容早期 chunked-direct checkpoint：temporal_agg + chunk_emb + chunk_mlp
                self.temporal_agg = nn.Linear(seq_len, 1, bias=False)
                self.chunk_emb = nn.Embedding(self._dp_n_chunks, d_model)
                self.future_time_proj = nn.Linear(self.time_feature_dim, d_model, bias=False) if self.use_future_time_features else None
                self.chunk_mlp = nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, self._dp_chunk_size * self.out_dim),
                )
            if self.direct_step_refine:
                # 零初始化的逐步 residual refiner：
                # 1) 兼容旧 checkpoint 的 model-only warm start
                # 2) 在不破坏旧 chunkwise 输出的前提下，补足 chunk 内逐步校准能力
                self.chunk_step_emb = nn.Embedding(self._dp_chunk_size, d_model)
                self.future_step_time_proj = nn.Linear(self.time_feature_dim, d_model, bias=False) if self.use_future_time_features else None
                self.step_refine = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, self.out_dim),
                )
                nn.init.zeros_(self.step_refine[-1].weight)
                nn.init.zeros_(self.step_refine[-1].bias)
        else:
            # 自回归解码器（原始结构）
            self.dec_step_emb = nn.Embedding(pred_len, d_model)
            dec_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=ff_multiplier * d_model,
                dropout=dropout, batch_first=False,
                activation="gelu", norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.dropout_layer = nn.Dropout(dropout)
        self.ln_embed = nn.LayerNorm(d_model)
        if self.gated_residual:
            gate_in_dim = d_model + (self.time_feature_dim if self.use_future_time_features else 0)
            self.residual_gate = nn.Sequential(
                nn.Linear(gate_in_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, self.out_dim),
            )
            nn.init.zeros_(self.residual_gate[-1].weight)
            nn.init.constant_(self.residual_gate[-1].bias, 3.0)

    @staticmethod
    def _gen_causal_mask(sz, device):
        # bool mask: True = masked out; enables Flash Attention (vs float -inf which forces slow O(N²) path)
        return torch.ones(sz, sz, device=device, dtype=torch.bool).triu(diagonal=1)

    @staticmethod
    def _precompute_khop(A_raw, k_hop):
        """预计算K-hop邻接矩阵列表。
        A_raw 在外部通常已经做过对称归一化，这里只做逐跳幂和逐行归一化，避免双重归一化。"""
        N = A_raw.shape[0]
        hops = [torch.eye(N, device=A_raw.device, dtype=A_raw.dtype)]  # k=0: I
        A_power = A_raw.clone()
        for k in range(1, k_hop + 1):
            # 行归一化（保证每跳的聚合权重和=1）
            row_sum = A_power.sum(dim=1, keepdim=True).clamp_min(1e-8)
            hops.append(A_power / row_sum)
            if k < k_hop:
                A_power = A_power @ A_raw
        return hops

    def _infer_future_time_features(self, x):
        """Infer raw future calendar features from observed TOD/DOW channels."""
        if self.time_feature_dim <= 0 or not self.use_future_time_features:
            return None
        time_feats = x[:, :, :, self.time_feature_start:self.time_feature_start + self.time_feature_dim]
        tod = time_feats[:, :, :, 0:1]  # [B, T, N, 1]
        last_tod = tod[:, -1:, :, :]
        if tod.shape[1] >= 2:
            prev_tod = tod[:, -2:-1, :, :]
            step = torch.remainder(last_tod - prev_tod, 1.0)
            fallback = torch.full_like(step, 1.0 / 288.0)
            step = torch.where(step.abs() < 1e-6, fallback, step)
        else:
            step = torch.full_like(last_tod, 1.0 / 288.0)
        horizon = torch.arange(1, self.pred_len + 1, device=x.device, dtype=x.dtype).view(1, self.pred_len, 1, 1)
        tod_linear = last_tod + step * horizon
        future = [torch.remainder(tod_linear, 1.0)]
        if self.time_feature_dim >= 2:
            dow = time_feats[:, :, :, 1:2]
            last_dow = dow[:, -1:, :, :]
            day_roll = torch.floor(tod_linear)
            future.append(torch.remainder(last_dow + day_roll / 7.0, 1.0))
        for idx in range(2, self.time_feature_dim):
            future.append(time_feats[:, -1:, :, idx:idx + 1].repeat(1, self.pred_len, 1, 1))
        return torch.cat(future, dim=-1)

    def _refine_direct_chunk(self, base_chunk, h_i, future_time, start, end):
        if not self.direct_step_refine:
            return base_chunk

        chunk_len = base_chunk.shape[2]
        step_ids = torch.arange(chunk_len, device=base_chunk.device)
        h_step = h_i.unsqueeze(2) + self.chunk_step_emb(step_ids).view(1, 1, chunk_len, self.d_model)
        if future_time is not None and self.future_step_time_proj is not None:
            time_step = future_time[:, start:end, :, :].permute(0, 2, 1, 3).contiguous()
            h_step = h_step + self.future_step_time_proj(time_step)
        return base_chunk + self.step_refine(h_step)

    def _apply_residual_delta(self, y_hat, x, h_enc, future_time):
        last_val = x[:, -1:, :, :self.out_dim]
        if not self.gated_residual:
            return y_hat + last_val.repeat(1, self.pred_len, 1, 1)

        h_last = h_enc[:, -1, :, :].unsqueeze(1).repeat(1, self.pred_len, 1, 1)
        if future_time is not None and self.time_feature_dim > 0:
            gate_input = torch.cat([h_last, future_time], dim=-1)
        else:
            gate_input = h_last
        gate = torch.sigmoid(self.residual_gate(gate_input))
        return y_hat + gate * last_val.repeat(1, self.pred_len, 1, 1)

    def forward(self, x, A, SF, TF, y_in=None):
        B, T_in, N, F_dim = x.shape
        device = x.device

        # 1) Fractal embeddings
        E_s = self.ln_sf(self.sf_proj(SF))
        E_t = self.ln_tf(self.tf_proj(TF))

        # 2) Bidirectional attention → 得到节点表征 + 空间注意力权重
        H_cat, S_s2t, S_t2s, B_st, B_ts = self.biattn(E_s, E_t, A)
        H_node = self.fuse(H_cat)  # [N, d]

        # 空间混合权重：根据模式选择
        S_attention = (S_s2t + S_t2s) / 2.0  # [N, N]
        if self.spatial_mode == "gcn":
            # GCN模式：使用多跳邻接矩阵
            if not hasattr(self, '_A_hops_cache') or self._A_hops_cache is None:
                self._A_hops_cache = self._precompute_khop(A, self.k_hop)
            S_spatial = self._A_hops_cache
        elif self.spatial_mode == "hybrid":
            if not hasattr(self, '_A_hops_cache') or self._A_hops_cache is None:
                self._A_hops_cache = self._precompute_khop(A, self.k_hop)
            S_spatial = (S_attention, self._A_hops_cache)
        else:
            # Attention模式（默认）：平均两个方向的注意力
            S_spatial = S_attention

        # 3) Encoder input
        x_emb = self.in_proj(x)                                       # [B, T, N, d]
        x_emb = self.pos_enc(x_emb)
        x_emb = x_emb + H_node.view(1, 1, N, self.d_model)
        x_emb = self.ln_embed(self.dropout_layer(x_emb))              # [B, T, N, d]

        # 4) 时空编码器：交替进行时间注意力和空间混合
        h_enc = x_emb
        for layer in self.st_encoder:
            h_enc = layer(h_enc, S_spatial)                            # [B, T, N, d]

        # 5) Predict
        future_time = self._infer_future_time_features(x)
        decoder_future_time = future_time if self.decoder_future_tod else None
        if self.use_direct_pred:
            # 直接预测路径（分组MLP）:
            # legacy: 单一 temporal_agg 压缩整段历史
            # chunkwise: 为每个未来 chunk 学独立的时间聚合，降低长步长信息损失
            h_time = h_enc.permute(0, 2, 3, 1)  # [B, N, d, T]
            if self.direct_head_mode == "chunkwise":
                h_chunk = self.temporal_chunk_proj(h_time).permute(0, 3, 1, 2).contiguous()  # [B, C, N, d]
                h_last = h_enc[:, -1, :, :]  # [B, N, d]
            elif self.direct_head_mode == "attn":
                h_key = self.temporal_key(h_enc)  # [B, T, N, d]
                h_last = h_enc[:, -1, :, :]       # [B, N, d]
            elif self.direct_head_mode == "linear":
                h_agg = self.temporal_agg(h_time).squeeze(-1)  # [B, N, d]
                h_base = F.gelu(self.out_hidden(h_agg))
                y_hat = self.out_proj_direct(h_base).view(B, N, self.pred_len, self.out_dim)
                y_hat = y_hat.permute(0, 2, 1, 3).contiguous()  # [B, pred_len, N, out_dim]
            else:
                h_agg = self.temporal_agg(h_time).squeeze(-1)  # [B, N, d]
                h_base = F.gelu(self.out_hidden(h_agg))        # [B, N, d]
            if self.direct_head_mode != "linear":
                chunks = []
                for i in range(self._dp_n_chunks):
                    start = i * self._dp_chunk_size
                    end = min((i + 1) * self._dp_chunk_size, self.pred_len)
                    if self.direct_head_mode == "chunkwise":
                        h_i = F.gelu(self.out_hidden(h_chunk[:, i, :, :] + h_last))
                    elif self.direct_head_mode == "attn":
                        q_i = self.chunk_query.weight[i].view(1, 1, 1, self.d_model)
                        score = (h_key * q_i).sum(dim=-1) / math.sqrt(self.d_model)  # [B, T, N]
                        weight = torch.softmax(score, dim=1)
                        context = torch.sum(weight.unsqueeze(-1) * h_enc, dim=1)     # [B, N, d]
                        h_i = F.gelu(self.out_hidden(context + h_last))
                    else:
                        h_i = h_base
                    h_i = h_i + self.chunk_emb.weight[i]  # [B, N, d]
                    if future_time is not None and self.future_time_proj is not None:
                        time_chunk = future_time[:, start:end, :, :].mean(dim=1)  # [B, N, Ft]
                        h_i = h_i + self.future_time_proj(time_chunk)
                    c_i = self.chunk_mlp(h_i).view(B, N, self._dp_chunk_size, self.out_dim)
                    if i == self._dp_n_chunks - 1 and self._dp_last_chunk != self._dp_chunk_size:
                        c_i = c_i[:, :, :self._dp_last_chunk, :]
                    c_i = self._refine_direct_chunk(c_i, h_i, future_time, start, end)
                    chunks.append(c_i.reshape(B, N, -1))
                y_hat = torch.cat(chunks, dim=-1)  # [B, N, pred_len * out_dim]
                y_hat = y_hat.view(B, N, self.pred_len, self.out_dim)
                y_hat = y_hat.permute(0, 2, 1, 3).contiguous()  # [B, pred_len, N, out_dim]
        else:
            # 自回归解码器路径（原始结构）
            mem = h_enc.permute(1, 0, 2, 3).contiguous().view(T_in, B * N, self.d_model)

            if y_in is not None:
                dec_in = self.in_proj(y_in)
            else:
                dec_seed = x[:, -1:, :, :].repeat(1, self.pred_len, 1, 1)
                if decoder_future_time is not None:
                    dec_seed = dec_seed.clone()
                    start = self.time_feature_start
                    end = start + self.time_feature_dim
                    dec_seed[:, :, :, start:end] = decoder_future_time
                dec_in = self.in_proj(dec_seed)
                step_ids = torch.arange(self.pred_len, device=device)
                dec_in = dec_in + self.dec_step_emb(step_ids).view(1, self.pred_len, 1, self.d_model)

            dec_in = self.pos_enc(dec_in)
            dec_in = dec_in + H_node.view(1, 1, N, self.d_model)
            dec_in = self.ln_embed(self.dropout_layer(dec_in))
            dec_in = dec_in.permute(1, 0, 2, 3).contiguous().view(self.pred_len, B * N, self.d_model)

            tgt_mask = self._gen_causal_mask(self.pred_len, device)
            dec_out = self.decoder(tgt=dec_in, memory=mem, tgt_mask=tgt_mask)

            h = F.gelu(self.out_hidden(dec_out))
            y_hat = self.out_proj(h).view(self.pred_len, B, N, self.out_dim).permute(1, 0, 2, 3).contiguous()

        # 6) 残差修正（两条路径共用）
        if self.residual_delta:
            y_hat = self._apply_residual_delta(y_hat, x, h_enc, future_time)

        aux = {
            "S_s2t": S_s2t, "S_t2s": S_t2s,
            "B_st": B_st, "B_ts": B_ts,
            "H_st_node": H_node, "E_s": E_s, "E_t": E_t,
            "lambda_st": F.softplus(self.biattn._lambda_st),
            "lambda_ts": F.softplus(self.biattn._lambda_ts),
        }
        return y_hat, aux
