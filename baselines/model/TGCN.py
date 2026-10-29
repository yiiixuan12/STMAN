# -*- coding: utf-8 -*-
# TGCN.py
# Stable and accelerated T-GCN: dense support + optional non-autoregressive output + strided encoding

from typing import Optional
import torch
import torch.nn as nn


# ------------------------------------------------------------
# Symmetrically normalized adjacency:Â = D^{-1/2}(A + I)D^{-1/2}
# ------------------------------------------------------------
def build_symmetric_normalized_adj(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """
    Input:
        adj: [N,N] Raw adjacency (may be asymmetric or non-binary)
        add_self_loops: Whether to add self-loops
    Returns:
        support: [N,N] Dense symmetrically normalized adjacency (contiguous, float32)
    """
    assert adj.dim() == 2 and adj.size(0) == adj.size(1), "adj must be a square matrix [N,N]"
    A = adj.clone().to(dtype=torch.float32)
    N = A.size(0)
    if add_self_loops:
        A = A + torch.eye(N, dtype=A.dtype, device=A.device)

    deg = A.sum(dim=1)                                 # A may be asymmetric; use row sums here
    deg_inv_sqrt = torch.pow(deg.clamp_min(1e-12), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    support = D_inv_sqrt @ A @ D_inv_sqrt
    return support.contiguous()


# ------------------------------------------------------------
# Graph convolution: left-multiply [x_t, h_{t-1}] by support, then apply a linear map
# ------------------------------------------------------------
class TGCNGraphConvolution(nn.Module):
    """
    Input:
        inputs      : [B, N, F_in]
        hidden_state: [B, N, H]
    Output:
        out         : [B, N, out_dim]
    Implementation notes:
        - Use a dense support matrix to avoid sparse/compilation instability
        - Reshape [B,N,C] -> [N,B*C] for one large matrix multiplication without a for loop
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
        assert inputs.dim() == 3, "inputs must have shape [B,N,F_in]"
        assert hidden_state.dim() == 3, "hidden_state must have shape [B,N,H]"
        B, N, _ = inputs.shape
        assert hidden_state.shape == (B, N, self.hidden_dim), \
            f"hidden_state must have shape [B,N,{self.hidden_dim}]"
        assert self.support.shape == (N, N), f"support must have shape [{N},{N}]"

        # Concatenate and diffuse
        xh = torch.cat([inputs, hidden_state], dim=-1).contiguous()  # [B,N,C]
        C = self.in_feat_dim + self.hidden_dim

        xh_2d  = xh.permute(1, 0, 2).reshape(N, B * C).contiguous()  # [N,B*C]
        axh_2d = self.support @ xh_2d                                 # [N,B*C]
        axh    = axh_2d.reshape(N, B, C).permute(1, 0, 2).contiguous()# [B,N,C]

        out = axh @ self.weights + self.biases                        # [B,N,out_dim]
        return out


# ------------------------------------------------------------
# T-GCN Cell: GCN-generated gates
# ------------------------------------------------------------
class TGCNCell(nn.Module):
    """
    GRU-like:
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
            bias_init=bias_for_gates,   # Positive bias helps retain memory
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
# Encoder: iterate the cell and return the final hidden state
# ------------------------------------------------------------
class TGCNEncoder(nn.Module):
    """
    Input: x ∈ [B, T_in, N, F_in]
    Output: h_T ∈ [B, N, H]
    """
    def __init__(self, support: torch.Tensor, in_feat_dim: int, hidden_dim: int):
        super().__init__()
        self.cell = TGCNCell(support, in_feat_dim, hidden_dim)

    def init_hidden(self, batch_size: int, num_nodes: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, num_nodes, self.cell.hidden_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "x must have shape [B,T,N,F]"
        B, T, N, _ = x.shape
        h = self.init_hidden(B, N, x.device)
        for t in range(T):
            h = self.cell(x[:, t, :, :].contiguous(), h)
        return h  # [B,N,H]


# ------------------------------------------------------------
# Forecaster: supports non-autoregressive output and strided encoding
# ------------------------------------------------------------
class TGCNForecaster(nn.Module):
    """
    Full T-GCN multi-step forecasting (accelerated for long sequences)
    - non_autoregressive=True: Output all H steps at once without an autoregressive decoding loop
    - enc_stride>=1: Downsample time steps during encoding (1=no downsampling; 4/8 provides substantial speedups)

    Input:  x ∈ [B, T_in, N, F_in]
    Output:  y ∈ [B, H, N, out_dim]
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
                 use_pred_for_next: bool = True,     # For backward compatibility; unused in non-autoregressive mode
                 map_pred_to_infeat_dims: int = 1,   # Same as above
                 non_autoregressive: bool = True,    # ✅ Output all steps at once
                 enc_stride: int = 1                 # ✅ Strided encoding (suggested: 2/4/8)
                 ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.horizon = int(horizon)
        self.non_autoregressive = bool(non_autoregressive)
        self.enc_stride = max(1, int(enc_stride))

        # Backward-compatible parameters (used only in the autoregressive path)
        self.use_pred_for_next = bool(use_pred_for_next)
        self.map_pred_to_infeat_dims = int(map_pred_to_infeat_dims)

        with torch.no_grad():
            support = build_symmetric_normalized_adj(adj.float(), add_self_loops=add_self_loops)

        self.encoder = TGCNEncoder(support, in_feat_dim=input_dim, hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if self.non_autoregressive:
            # Project the hidden state directly to (H*out_dim)
            self.readout_all = nn.Linear(hidden_dim, out_dim * horizon)
        else:
            # For stepwise decoding
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
        Sample x with enc_stride before passing it to Encoder to reduce time-step iterations.
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
        assert x.dim() == 4, "x must have shape [B,T_in,N,F_in]"
        B, T_in, N, F_in = x.shape
        assert N == self.num_nodes and F_in == self.input_dim, \
            f"Input dimensions do not match the model: N={N}/{self.num_nodes}, F_in={F_in}/{self.input_dim}"

        # 1) Encode (with optional strided sampling)
        h = self._encode_with_stride(x)        # [B,N,H]
        h = self.dropout(h)

        if self.non_autoregressive:
            # 2) Output all H steps at once
            y_all = self.readout_all(h)                     # [B,N,H*out_dim]
            y_all = y_all.view(B, N, self.horizon, self.out_dim)
            y_all = y_all.permute(0, 2, 1, 3).contiguous()  # [B,H,N,out_dim]
            return y_all

        # ===== Compatibility: autoregressive path (set non_autoregressive=False to use)=====
        dec_in = self._prepare_decode_input(x[:, -1, :, :])  # [B,N,F_in]
        preds = []
        for t in range(self.horizon):
            h = self.encoder.cell(dec_in, h)                 # [B,N,H]
            y_t = self.readout_step(h)                       # [B,N,out_dim]
            preds.append(y_t)

            # Next decoding input
            if self.use_pred_for_next and self.pred2dec is not None:
                dec_feedback = self.pred2dec(y_t)            # [B,N,k]
                dec_in_next = dec_in.clone()
                k = self.map_pred_to_infeat_dims
                dec_in_next[:, :, :k] = dec_feedback
            else:
                dec_in_next = dec_in

            # Teacher forcing (optional)
            if (y_teacher is not None) and (teacher_forcing_ratio > 0.0):
                if torch.rand(1, device=x.device).item() < float(teacher_forcing_ratio):
                    if self.pred2dec is not None and self.map_pred_to_infeat_dims > 0:
                        k = self.map_pred_to_infeat_dims
                        true_feedback = self.pred2dec(y_teacher[:, t])
                        dec_in_next[:, :, :k] = true_feedback

            dec_in = dec_in_next

        y = torch.stack(preds, dim=1)                        # [B,H,N,out_dim]
        return y
