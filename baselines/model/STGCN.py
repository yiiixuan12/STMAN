# STGCN实现
# -*- coding: utf-8 -*-
"""
输入张量形状: [B, C_in, T, N]

需要的 args 字段：
- n_his: 历史时间步数 (int)
- Kt: 时域卷积核长 (int)
- Ks: 图卷积阶数(cheb)或占位(graph) (int)
- act_func: 'glu' | 'gtu' | 'relu' | 'silu'
- graph_conv_type: 'cheb_graph_conv' | 'graph_conv'
- gso: 图移位算子 (torch.FloatTensor[N, N])；Cheb 版需已缩放到[-1,1]
- enable_bias: 线性层/图卷积是否带偏置 (bool)
- droprate: Dropout 概率 (float)

blocks 说明（与常见 STGCN 配置兼容）：
例如 blocks = [
    [C_in],                 # 输入通道占位（只用到最后一个数）
    [64, 16, 64],           # ST-Block1: (T:64 -> G:16 -> T:64)
    [64, 16, 64],           # ST-Block2
    [64],                   # OutputBlock 输入通道（上一层输出通道）
    [128],                  # OutputBlock 中间全连接通道
    [C_out]                 # 输出通道
]
其中 ST-Blocks 的数量 = len(blocks) - 3
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# -----------------------------
# 基础组件
# -----------------------------
class Align(nn.Module):
    """通过 1x1 卷积或零填充把通道对齐到指定维度"""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            b, _, t, n = x.shape
            pad = x.new_zeros(b, self.c_out - self.c_in, t, n)
            x = torch.cat([x, pad], dim=1)
        return x


class CausalConv1d(nn.Conv1d):
    """一维因果卷积（可选左填充）"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 enable_padding=False, dilation=1, groups=1, bias=True):
        self._padding = (kernel_size - 1) * dilation if enable_padding else 0
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=self._padding,
                         dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        y = super().forward(x)
        return y[:, :, : -self._padding] if self._padding != 0 else y


class CausalConv2d(nn.Conv2d):
    """二维因果卷积：仅在时间维(T)做左填充，节点维(N)不填充"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 enable_padding=False, dilation=1, groups=1, bias=True):
        k = nn.modules.utils._pair(kernel_size)
        s = nn.modules.utils._pair(stride)
        d = nn.modules.utils._pair(dilation)
        if enable_padding:
            self._pad_t = int((k[0] - 1) * d[0])
            self._pad_n = 0
        else:
            self._pad_t = 0
            self._pad_n = 0
        super().__init__(in_channels, out_channels, kernel_size=k, stride=s,
                         padding=0, dilation=d, groups=groups, bias=bias)

    def forward(self, x):
        if self._pad_t or self._pad_n:
            x = F.pad(x, (self._pad_n, 0, self._pad_t, 0))  # (N_left,N_right,T_left,T_right)
        return super().forward(x)


class TemporalConvLayer(nn.Module):
    """
    带门控的时域卷积（GLU/GTU），保持因果性。
    输入:  [B, C_in, T, N]
    输出:  [B, C_out, T - (Kt-1), N] （未启用padding时）
    """
    def __init__(self, Kt: int, c_in: int, c_out: int, n_vertex: int, act_func: str):
        super().__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)

        if act_func in ('glu', 'gtu'):
            out_c = 2 * c_out
        else:
            out_c = c_out

        # 因果卷积：默认不做padding，时间长度将缩短 Kt-1
        self.causal_conv = CausalConv2d(
            in_channels=c_in, out_channels=out_c,
            kernel_size=(Kt, 1), enable_padding=False, dilation=1
        )
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.act_func = act_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对齐残差分支，并裁剪时间步与主分支对齐
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal = self.causal_conv(x)

        if self.act_func in ('glu', 'gtu'):
            x_p, x_q = x_causal[:, : self.c_out], x_causal[:, -self.c_out:]
            if self.act_func == 'glu':
                x = (x_p + x_in) * torch.sigmoid(x_q)
            else:  # gtu
                x = torch.tanh(x_p + x_in) * torch.sigmoid(x_q)
        elif self.act_func == 'relu':
            x = self.relu(x_causal + x_in)
        elif self.act_func == 'silu':
            x = self.silu(x_causal + x_in)
        else:
            raise NotImplementedError(f'Unknown act_func: {self.act_func}')
        return x


# -----------------------------
# 图卷积（Cheb / GCN）
# -----------------------------
class ChebGraphConv(nn.Module):
    """
    Chebyshev 多项式图卷积
    输入:  x [B, C_in, T, N]
    输出:      [B, T, N, C_out]
    要求：gso 已缩放至 [-1, 1]
    """
    def __init__(self, c_in: int, c_out: int, Ks: int, gso: torch.Tensor, bias: bool):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso  # [N, N]
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, T, N] -> [B, T, N, C]
        x = x.permute(0, 2, 3, 1)

        if self.Ks <= 0:
            raise ValueError(f'Ks must be positive, got {self.Ks}')

        N = self.gso.size(0)

        # T_0(x) = x
        x_list = [x]
        if self.Ks >= 2:
            # T_1(x) = gso @ x  （在节点维做乘法）
            x1 = torch.einsum('nm,btnc->btmc', self.gso, x)
            x_list.append(x1)
        for k in range(2, self.Ks):
            x_k = torch.einsum('nm,btnc->btmc', 2 * self.gso, x_list[-1]) - x_list[-2]
            x_list.append(x_k)

        # [B,T,K,N,C_in]
        xk = torch.stack(x_list, dim=2)
        # 线性组合： [B,T,K,N,C_in] × [K,C_in,C_out] -> [B,T,N,C_out]
        y = torch.einsum('btknc,kcd->btnd', xk, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class GraphConv(nn.Module):
    """
    GCN 风格的一阶图卷积（采用重归一化技巧），本实现假设 gso 已含该处理
    输入:  x [B, C_in, T, N]
    输出:      [B, T, N, C_out]
    """
    def __init__(self, c_in: int, c_out: int, gso: torch.Tensor, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out)) if bias else None
        self.gso = gso
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, T, N] -> [B, T, N, C]
        x = x.permute(0, 2, 3, 1)                      # [B,T,N,C_in]
        x = torch.einsum('nm,btnc->btmc', self.gso, x) # 邻接聚合 [B,T,N,C_in]
        y = torch.einsum('btnc,cd->btnd', x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y


class GraphConvLayer(nn.Module):
    """对齐通道后施加图卷积，并做残差（与对齐前后的通道一致）"""
    def __init__(self, graph_conv_type: str, c_in: int, c_out: int, Ks: int, gso: torch.Tensor, bias: bool):
        super().__init__()
        self.graph_conv_type = graph_conv_type
        self.align = Align(c_in, c_out)
        if graph_conv_type == 'cheb_graph_conv':
            self.gc = ChebGraphConv(c_out, c_out, Ks, gso, bias)
        elif graph_conv_type == 'graph_conv':
            self.gc = GraphConv(c_out, c_out, gso, bias)
        else:
            raise NotImplementedError(f'Unknown graph_conv_type: {graph_conv_type}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.align(x)                           # [B, c_out, T, N]
        y = self.gc(x_in)                              # [B, T, N, c_out]
        y = y.permute(0, 3, 1, 2)                      # [B, c_out, T, N]
        return y + x_in


# -----------------------------
# ST-Block 与 输出块
# -----------------------------
class STConvBlock(nn.Module):
    """
    结构: T (GLU/GTU) -> G (Cheb/GCN) -> T -> LayerNorm -> Dropout
    """
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels,
                 act_func, graph_conv_type, gso, bias, droprate):
        super().__init__()
        # channels: [T1_out, G_out, T2_out]
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
        return x


class OutputBlock(nn.Module):
    """结构: T -> LayerNorm -> FC -> ReLU -> Dropout -> FC"""
    def __init__(self, Ko, last_block_channel, channels, end_channel,
                 n_vertex, act_func, bias, droprate):
        super().__init__()
        # channels: [T_out, FC_hidden]
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], n_vertex, act_func)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]], eps=1e-12)
        self.fc1 = nn.Linear(channels[0], channels[1], bias=bias)
        self.fc2 = nn.Linear(channels[1], end_channel, bias=bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tmp_conv1(x)                          # [B, C, T', N]
        x = self.tc1_ln(x.permute(0, 2, 3, 1))         # [B, T', N, C]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).permute(0, 3, 1, 2)            # [B, C_out, T', N]
        return x


# -----------------------------
# 顶层模型（Cheb / GCN 两版）
# -----------------------------
class STGCNChebGraphConv(nn.Module):
    """包含结构:  (T-G-T-N-D) × L  +  (T-N-F-F)"""
    def __init__(self, args, blocks, n_vertex):
        super().__init__()
        modules = []
        # ST-Blocks
        for l in range(len(blocks) - 3):
            modules.append(
                STConvBlock(args.Kt, args.Ks, n_vertex,
                            blocks[l][-1], blocks[l + 1],
                            args.act_func, args.graph_conv_type,
                            args.gso, args.enable_bias, args.droprate)
            )
        self.st_blocks = nn.Sequential(*modules)

        # 计算 OutputBlock 的时域核长 Ko（未做 padding 情况下）
        self.Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        if self.Ko >= 1:
            self.output = OutputBlock(self.Ko, blocks[-3][-1], blocks[-2], blocks[-1][0],
                                      n_vertex, args.act_func, args.enable_bias, args.droprate)
        else:  # Ko == 0
            self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.st_blocks(x)
        if self.Ko >= 1:
            x = self.output(x)
        else:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        return x


class STGCNGraphConv(nn.Module):
    """包含结构:  (T-G-T-N-D) × L  +  (T-N-F-F)（图卷积为 GCN 版）"""
    def __init__(self, args, blocks, n_vertex):
        super().__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(
                STConvBlock(args.Kt, args.Ks, n_vertex,
                            blocks[l][-1], blocks[l + 1],
                            args.act_func, args.graph_conv_type,
                            args.gso, args.enable_bias, args.droprate)
            )
        self.st_blocks = nn.Sequential(*modules)

        self.Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        if self.Ko >= 1:
            self.output = OutputBlock(self.Ko, blocks[-3][-1], blocks[-2], blocks[-1][0],
                                      n_vertex, args.act_func, args.enable_bias, args.droprate)
        else:
            self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.st_blocks(x)
        if self.Ko >= 1:
            x = self.output(x)
        else:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        return x
