# -*- coding: utf-8 -*-
# TGCN.py
# 稳定&加速版 T-GCN：稠密 support + 可选非自回归输出 + 编码步长采样

from typing import Optional
import torch
import torch.nn as nn


# ------------------------------------------------------------
# 对称归一化邻接：Â = D^{-1/2}(A + I)D^{-1/2}
# ------------------------------------------------------------
def build_symmetric_normalized_adj(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """
    输入:
        adj: [N,N] 原始邻接（允许非对称/非二值）
        add_self_loops: 是否加自环
    返回:
        support: [N,N] 稠密对称归一化邻接（contiguous，float32）
    """
    assert adj.dim() == 2 and adj.size(0) == adj.size(1), "adj 必须是方阵 [N,N]"
    A = adj.clone().to(dtype=torch.float32)
    N = A.size(0)
    if add_self_loops:
        A = A + torch.eye(N, dtype=A.dtype, device=A.device)

    deg = A.sum(dim=1)                                 # 允许 A 非对称，这里用行和
    deg_inv_sqrt = torch.pow(deg.clamp_min(1e-12), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    support = D_inv_sqrt @ A @ D_inv_sqrt
    return support.contiguous()


# ------------------------------------------------------------
# 图卷积：对 [x_t, h_{t-1}] 做 support 左乘，再线性映射
# ------------------------------------------------------------
class TGCNGraphConvolution(nn.Module):
    """
    输入:
        inputs      : [B, N, F_in]
        hidden_state: [B, N, H]
    输出:
        out         : [B, N, out_dim]
    实现要点：
        - support 使用稠密矩阵，避免稀疏/编译带来的不稳定
        - 将 [B,N,C] → [N,B*C] 做一次性大矩阵乘，避免 for 循环
    """
    def __init__(self,
                 support: torch.Tensor,
                 in_feat_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 bias_init: float = 0.0):
        super().__init__()
        self.in_feat_dim = int(in_feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        if support.is_sparse:
            support = support.to_dense()
        self.register_buffer("support", support.contiguous())  # [N,N]

        self.weights = nn.Parameter(torch.empty(self.in_feat_dim + self.hidden_dim, self.output_dim))
        self.biases  = nn.Parameter(torch.empty(self.output_dim))
        self.reset_parameters(bias_init)

    def reset_parameters(self, bias_init: float = 0.0):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, bias_init)

    def forward(self, inputs: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        assert inputs.dim() == 3, "inputs 需为 [B,N,F_in]"
        assert hidden_state.dim() == 3, "hidden_state 需为 [B,N,H]"
        B, N, _ = inputs.shape
        assert hidden_state.shape == (B, N, self.hidden_dim), \
            f"hidden_state 需为 [B,N,{self.hidden_dim}]"
        assert self.support.shape == (N, N), f"support 大小需为 [{N},{N}]"

        # 拼接与扩散
        xh = torch.cat([inputs, hidden_state], dim=-1).contiguous()  # [B,N,C]
        C = self.in_feat_dim + self.hidden_dim

        xh_2d  = xh.permute(1, 0, 2).reshape(N, B * C).contiguous()  # [N,B*C]
        axh_2d = self.support @ xh_2d                                 # [N,B*C]
        axh    = axh_2d.reshape(N, B, C).permute(1, 0, 2).contiguous()# [B,N,C]

        out = axh @ self.weights + self.biases                        # [B,N,out_dim]
        return out


# ------------------------------------------------------------
# T-GCN Cell：GCN 生成门控
# ------------------------------------------------------------
class TGCNCell(nn.Module):
    """
    GRU-like：
        [r, u] = sigmoid( GCN([x, h]) )
        c      = tanh(    GCN([x, r*h]) )
        h'     = u*h + (1-u)*c
    """
    def __init__(self,
                 support: torch.Tensor,
                 in_feat_dim: int,
                 hidden_dim: int,
                 bias_for_gates: float = 1.0):
        super().__init__()
        self.in_feat_dim = int(in_feat_dim)
        self.hidden_dim = int(hidden_dim)

        self.gconv_gates = TGCNGraphConvolution(
            support=support,
            in_feat_dim=in_feat_dim,
            hidden_dim=hidden_dim,
            output_dim=2 * hidden_dim,
            bias_init=bias_for_gates,   # 正偏置有助记忆
        )
        self.gconv_cand = TGCNGraphConvolution(
            support=support,
            in_feat_dim=in_feat_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            bias_init=0.0,
        )

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gconv_gates(x_t, h_prev))     # [B,N,2H]
        r, u = torch.chunk(gates, chunks=2, dim=-1)              # [B,N,H],[B,N,H]
        c = torch.tanh(self.gconv_cand(x_t, r * h_prev))         # [B,N,H]
        h = u * h_prev + (1.0 - u) * c                           # [B,N,H]
        return h


# ------------------------------------------------------------
# 编码器：循环 cell，输出最后隐状态
# ------------------------------------------------------------
class TGCNEncoder(nn.Module):
    """
    输入: x ∈ [B, T_in, N, F_in]
    输出: h_T ∈ [B, N, H]
    """
    def __init__(self, support: torch.Tensor, in_feat_dim: int, hidden_dim: int):
        super().__init__()
        self.cell = TGCNCell(support, in_feat_dim, hidden_dim)

    def init_hidden(self, batch_size: int, num_nodes: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, num_nodes, self.cell.hidden_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "x 需为 [B,T,N,F]"
        B, T, N, _ = x.shape
        h = self.init_hidden(B, N, x.device)
        for t in range(T):
            h = self.cell(x[:, t, :, :].contiguous(), h)
        return h  # [B,N,H]


# ------------------------------------------------------------
# 预测器：支持非自回归 & 编码步长采样
# ------------------------------------------------------------
class TGCNForecaster(nn.Module):
    """
    完整 T-GCN 多步预测（长序列提速版）
    - non_autoregressive=True: 一次性输出所有 H 步，去掉自回归解码循环
    - enc_stride>=1: 编码阶段的时间步降采样（1=不降采样；4/8 显著加速）

    输入:  x ∈ [B, T_in, N, F_in]
    输出:  y ∈ [B, H, N, out_dim]
    """
    def __init__(self,
                 adj: torch.Tensor,
                 num_nodes: int,
                 input_dim: int = 1,
                 hidden_dim: int = 64,
                 out_dim: int = 1,
                 horizon: int = 12,
                 add_self_loops: bool = True,
                 dropout: float = 0.0,
                 use_pred_for_next: bool = True,     # 为兼容旧接口，非自回归时不会用到
                 map_pred_to_infeat_dims: int = 1,   # 同上
                 non_autoregressive: bool = True,    # ✅ 一次性输出全部步
                 enc_stride: int = 1                 # ✅ 编码步长采样（建议 2/4/8）
                 ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.horizon = int(horizon)
        self.non_autoregressive = bool(non_autoregressive)
        self.enc_stride = max(1, int(enc_stride))

        # 兼容旧参数（仅自回归路径使用）
        self.use_pred_for_next = bool(use_pred_for_next)
        self.map_pred_to_infeat_dims = int(map_pred_to_infeat_dims)

        with torch.no_grad():
            support = build_symmetric_normalized_adj(adj.float(), add_self_loops=add_self_loops)

        self.encoder = TGCNEncoder(support, in_feat_dim=input_dim, hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if self.non_autoregressive:
            # 直接把隐藏状态投到 (H*out_dim)
            self.readout_all = nn.Linear(hidden_dim, out_dim * horizon)
        else:
            # 逐步解码用
            self.readout_step = nn.Linear(hidden_dim, out_dim)
            if self.use_pred_for_next and self.map_pred_to_infeat_dims > 0:
                self.pred2dec = nn.Linear(out_dim, self.map_pred_to_infeat_dims, bias=False)
            else:
                self.pred2dec = None

    @torch.no_grad()
    def _prepare_decode_input(self, last_obs: torch.Tensor) -> torch.Tensor:
        return last_obs.clone()  # [B,N,F_in]

    def _encode_with_stride(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入 x 以 enc_stride 采样后送入 Encoder，减少时间步循环次数。
        x: [B, T_in, N, F]
        """
        if self.enc_stride == 1:
            return self.encoder(x)  # [B,N,H]
        x_ds = x[:, ::self.enc_stride, :, :].contiguous()
        return self.encoder(x_ds)  # [B,N,H]

    def forward(self,
                x: torch.Tensor,                          # [B,T_in,N,F_in]
                y_teacher: Optional[torch.Tensor] = None, # [B,H,N,out_dim]
                teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        assert x.dim() == 4, "x 需为 [B,T_in,N,F_in]"
        B, T_in, N, F_in = x.shape
        assert N == self.num_nodes and F_in == self.input_dim, \
            f"输入维度与模型不一致: N={N}/{self.num_nodes}, F_in={F_in}/{self.input_dim}"

        # 1) 编码（可步长采样）
        h = self._encode_with_stride(x)        # [B,N,H]
        h = self.dropout(h)

        if self.non_autoregressive:
            # 2) 一次性输出全部 H 步
            y_all = self.readout_all(h)                     # [B,N,H*out_dim]
            y_all = y_all.view(B, N, self.horizon, self.out_dim)
            y_all = y_all.permute(0, 2, 1, 3).contiguous()  # [B,H,N,out_dim]
            return y_all

        # ===== 兼容：自回归路径（如需设置 non_autoregressive=False）=====
        dec_in = self._prepare_decode_input(x[:, -1, :, :])  # [B,N,F_in]
        preds = []
        for t in range(self.horizon):
            h = self.encoder.cell(dec_in, h)                 # [B,N,H]
            y_t = self.readout_step(h)                       # [B,N,out_dim]
            preds.append(y_t)

            # 下一步解码输入
            if self.use_pred_for_next and self.pred2dec is not None:
                dec_feedback = self.pred2dec(y_t)            # [B,N,k]
                dec_in_next = dec_in.clone()
                k = self.map_pred_to_infeat_dims
                dec_in_next[:, :, :k] = dec_feedback
            else:
                dec_in_next = dec_in

            # teacher forcing（可选）
            if (y_teacher is not None) and (teacher_forcing_ratio > 0.0):
                if torch.rand(1, device=x.device).item() < float(teacher_forcing_ratio):
                    if self.pred2dec is not None and self.map_pred_to_infeat_dims > 0:
                        k = self.map_pred_to_infeat_dims
                        true_feedback = self.pred2dec(y_teacher[:, t])
                        dec_in_next[:, :, :k] = true_feedback

            dec_in = dec_in_next

        y = torch.stack(preds, dim=1)                        # [B,H,N,out_dim]
        return y
