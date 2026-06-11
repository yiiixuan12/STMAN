# model.py
"""
MultiscaleMLP (single-file)
- MultiLayerPerceptron
- PatchEncoder
- DownsampEncoder
- MultiscaleMLP

Key fixes:
1) Use Conv1x1Auto (lazy 1x1 conv) for data_embedding_layer & projection1 to avoid in_channels mismatch:
   - In PatchEncoder: projection1 must see channels = D*P + d_td + d_dw (P = #patches = input_len/patch_len)
   - In DownsampEncoder: data embedding channels depend on actual L (after stride sampling), not patch_len
2) Safe handling when optional embeddings are disabled by using zero-width feature tensors that keep cat() happy.
"""

from math import ceil
import torch
from torch import nn
from einops import rearrange


# ---------- Small helper: lazy 1x1 conv that infers in_channels on first forward ----------
class Conv1x1Auto(nn.Module):
    def __init__(self, out_channels: int, bias: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.bias = bias
        self.conv = None  # will be created at first forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            in_ch = x.shape[1]
            self.conv = nn.Conv2d(in_ch, self.out_channels, kernel_size=1, bias=self.bias)
            # match dtype/device of input
            self.conv.to(dtype=x.dtype, device=x.device)
        return self.conv(x)


# ---------- Basic MLP block used in encoders ----------
class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links (1x1 convs)."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        # input_data: [B, D, P, N]
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        # residual + LayerNorm over channel dim
        hidden = self.norm((hidden + input_data).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return hidden


# ---------- Patch encoder ----------
class PatchEncoder(nn.Module):
    def __init__(self, td_size, td_codebook, dw_codebook, spa_codebook,
                 if_time_in_day, if_day_in_week, if_spatial,
                 input_dim, patch_len, stride, d_d, d_td, d_dw, d_spa, output_len, num_layer):
        super().__init__()
        self.td_codebook = td_codebook
        self.dw_codebook = dw_codebook
        self.spa_codebook = spa_codebook
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.if_spatial = if_spatial
        self.output_len = output_len
        self.td_size = td_size
        self.stride = stride
        self.patch_len = patch_len

        # Data embedding (channels depend on input_dim * patch_len): use lazy 1x1 conv
        self.data_embedding_layer = Conv1x1Auto(out_channels=d_d, bias=True)

        # hidden feature dim flowing through encoders
        self.hidden_dim = d_d + d_dw * int(self.if_day_in_week) * 2 + d_td * int(self.if_time_in_day) * 2

        self.temporal_encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim + d_spa * int(self.if_spatial),
                                   self.hidden_dim + d_spa * int(self.if_spatial)) for _ in range(num_layer)]
        )

        self.spatial_encoder = nn.Sequential(
            *[MultiLayerPerceptron(d_d + d_spa * int(self.if_spatial),
                                   d_d + d_spa * int(self.if_spatial)) for _ in range(num_layer)]
        )

        self.data_encoder = nn.Sequential(
            *[MultiLayerPerceptron(d_d, d_d) for _ in range(num_layer)]
        )

        # Projection to horizon T: channel count varies with #patches => lazy 1x1 conv
        self.projection1 = Conv1x1Auto(out_channels=output_len, bias=True)

    @staticmethod
    def _zero_feat_BP_N_D(B, P, N, device, D=0):
        return torch.zeros(B, P, N, D, device=device)

    @staticmethod
    def _zero_feat_BDN1(B, N, device, D=0):
        return torch.zeros(B, D, N, 1, device=device)

    def forward(self, patch_input):
        # patch_input: [B, P, L, N, C] where L == patch_len
        B, P, L, N, C = patch_input.shape
        device = patch_input.device

        # --- Temporal embeddings (start/end for each patch) ---
        # day-of-week
        if self.if_day_in_week:
            dow = patch_input[..., 2]  # [B, P, L, N]
            day_start = self.dw_codebook[(dow[:, :, 0, :]).to(torch.long)]       # [B, P, N, d_dw]
            day_end   = self.dw_codebook[(dow[:, :, -1, :]).to(torch.long)]      # [B, P, N, d_dw]
            future_day = day_end[:, -1, :, :].permute(0, 2, 1).unsqueeze(-1)     # [B, d_dw, N, 1]
        else:
            day_start = self._zero_feat_BP_N_D(B, P, N, device, D=0)
            day_end   = self._zero_feat_BP_N_D(B, P, N, device, D=0)
            future_day = self._zero_feat_BDN1(B, N, device, D=0)

        # time-of-day
        if self.if_time_in_day:
            tid = patch_input[..., 1]  # [B, P, L, N] (float in [0,1))
            tid_start = self.td_codebook[(tid[:, :, 0, :] * self.td_size).to(torch.long)]      # [B,P,N,d_td]
            tid_end   = self.td_codebook[(tid[:, :, -1, :] * self.td_size).to(torch.long)]     # [B,P,N,d_td]
            future_tid = self.td_codebook[(((tid[:, -1, -1, :] * self.td_size) + self.output_len) % self.td_size).to(torch.long)] \
                           .permute(0, 2, 1).unsqueeze(-1)  # [B, d_td, N, 1]
        else:
            tid_start = self._zero_feat_BP_N_D(B, P, N, device, D=0)
            tid_end   = self._zero_feat_BP_N_D(B, P, N, device, D=0)
            future_tid = self._zero_feat_BDN1(B, N, device, D=0)

        # --- Spatial embedding ---
        if self.if_spatial:
            # spa_codebook: [N, d_spa] -> [B, P, N, d_spa]
            spatial = self.spa_codebook.unsqueeze(0).expand(B, -1, -1).unsqueeze(1).expand(-1, P, -1, -1)
        else:
            spatial = None  # will be skipped in cat

        # --- Time series data embedding ---
        # Concatenate 3 channels along time dim => in_channels for 1x1 conv is (input_dim * L)
        data_cat = torch.cat((patch_input[..., 0], patch_input[..., 1], patch_input[..., 2]), dim=2)  # [B,P, 3L, N]
        data_emb = self.data_embedding_layer(data_cat.permute(0, 2, 1, 3))  # [B, d_d, P, N]
        data_emb = self.data_encoder(data_emb).permute(0, 2, 3, 1)         # [B, P, N, d_d]

        # --- Spatial encoder ---
        if spatial is not None:
            hidden = torch.cat((data_emb, spatial), dim=-1)  # [B, P, N, d_d + d_spa]
        else:
            hidden = data_emb                                 # [B, P, N, d_d]
        hidden = self.spatial_encoder(hidden.permute(0, 3, 1, 2))          # [B, D, P, N]
        hidden = hidden.permute(0, 2, 3, 1)                                 # [B, P, N, D]

        # --- Temporal encoder ---
        hidden = torch.cat((tid_start, day_start, hidden, tid_end, day_end), dim=-1)  # [B,P,N, D']
        hidden = self.temporal_encoder(hidden.permute(0, 3, 1, 2))                    # [B, D', P, N]

        # --- Flatten patches along channel, then append future (tid & dow), then project to T ---
        hidden = rearrange(hidden, 'B D P N -> B (D P) N').unsqueeze(-1)   # [B, D*P, N, 1]
        hidden = torch.cat((hidden, future_tid, future_day), dim=1)        # [B, D*P + d_td + d_dw, N, 1]
        predict = self.projection1(hidden)                                  # [B, T, N, 1]
        return predict


# ---------- Downsampled encoder ----------
class DownsampEncoder(nn.Module):
    def __init__(self, td_size, td_codebook, dw_codebook, spa_codebook,
                 if_time_in_day, if_day_in_week, if_spatial,
                 input_dim, patch_len, stride, d_d, d_td, d_dw, d_spa, output_len, num_layer):
        super().__init__()
        self.td_codebook = td_codebook
        self.dw_codebook = dw_codebook
        self.spa_codebook = spa_codebook
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.if_spatial = if_spatial
        self.output_len = output_len
        self.td_size = td_size
        self.stride = stride

        # Here L' (after stride sampling) != patch_len; use lazy 1x1 conv
        self.data_embedding_layer = Conv1x1Auto(out_channels=d_d, bias=True)

        self.hidden_dim = d_d + d_dw * int(self.if_day_in_week) * 2 + d_td * int(self.if_time_in_day) * 2

        self.temporal_encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim + d_spa * int(self.if_spatial),
                                   self.hidden_dim + d_spa * int(self.if_spatial)) for _ in range(num_layer)]
        )
        self.spatial_encoder = nn.Sequential(
            *[MultiLayerPerceptron(d_d + d_spa * int(self.if_spatial),
                                   d_d + d_spa * int(self.if_spatial)) for _ in range(num_layer)]
        )
        self.data_encoder = nn.Sequential(
            *[MultiLayerPerceptron(d_d, d_d) for _ in range(num_layer)]
        )

        # Safer to use lazy projection as well (channel count depends on stride concat + future emb)
        self.projection1 = Conv1x1Auto(out_channels=output_len, bias=True)

    @staticmethod
    def _zero_feat_BP_N_D(B, P, N, device, D=0):
        return torch.zeros(B, P, N, D, device=device)

    @staticmethod
    def _zero_feat_BDN1(B, N, device, D=0):
        return torch.zeros(B, D, N, 1, device=device)

    def forward(self, patch_input):
        # patch_input: [B, P, L', N, C] where P == stride
        B, P, Lp, N, C = patch_input.shape
        device = patch_input.device

        # time-of-day (start/end + future)
        if self.if_time_in_day:
            tid = patch_input[..., 1]  # [B, P, L', N]
            tid_start = self.td_codebook[(tid[:, :, 0, :] * self.td_size).to(torch.long)]
            tid_end   = self.td_codebook[(tid[:, :, -1, :] * self.td_size).to(torch.long)]
            future_tid = self.td_codebook[(((tid[:, -1, -1, :] * self.td_size) + self.output_len) % self.td_size).to(torch.long)] \
                           .permute(0, 2, 1).unsqueeze(-1)  # [B, d_td, N, 1]
        else:
            tid_start = self._zero_feat_BP_N_D(B, P, N, device, D=0)
            tid_end   = self._zero_feat_BP_N_D(B, P, N, device, D=0)
            future_tid = self._zero_feat_BDN1(B, N, device, D=0)

        # day-of-week (start/end + future)
        if self.if_day_in_week:
            dow = patch_input[..., 2]
            day_start = self.dw_codebook[(dow[:, :, 0, :]).to(torch.long)]
            day_end   = self.dw_codebook[(dow[:, :, -1, :]).to(torch.long)]
            future_day = day_end[:, -1, :, :].permute(0, 2, 1).unsqueeze(-1)  # [B, d_dw, N, 1]
        else:
            day_start = self._zero_feat_BP_N_D(B, P, N, device, D=0)
            day_end   = self._zero_feat_BP_N_D(B, P, N, device, D=0)
            future_day = self._zero_feat_BDN1(B, N, device, D=0)

        # spatial embedding
        if self.if_spatial:
            spatial = self.spa_codebook.unsqueeze(0).expand(B, -1, -1).unsqueeze(1).expand(-1, P, -1, -1)  # [B,P,N,d_spa]
        else:
            spatial = None

        # data embedding: concat along time dim for conv channel
        data_cat = torch.cat((patch_input[..., 0], patch_input[..., 1], patch_input[..., 2]), dim=2)  # [B,P, 3L', N]
        data_emb = self.data_embedding_layer(data_cat.permute(0, 2, 1, 3))  # [B, d_d, P, N]
        data_emb = self.data_encoder(data_emb).permute(0, 2, 3, 1)          # [B, P, N, d_d]

        # spatial encoder
        if spatial is not None:
            hidden = torch.cat((data_emb, spatial), dim=-1)  # [B,P,N, d_d + d_spa]
        else:
            hidden = data_emb
        hidden = self.spatial_encoder(hidden.permute(0, 3, 1, 2))  # [B, D, P, N]
        hidden = hidden.permute(0, 2, 3, 1)                        # [B, P, N, D]

        # temporal encoder
        hidden = torch.cat((tid_start, day_start, hidden, tid_end, day_end), dim=-1)  # [B,P,N,D']
        hidden = self.temporal_encoder(hidden.permute(0, 3, 1, 2))                    # [B, D', P, N]

        hidden = rearrange(hidden, 'B D P N -> B (D P) N').unsqueeze(-1)              # [B, D'*P, N, 1]
        hidden = torch.cat((hidden, future_tid, future_day), dim=1)                   # [B, ..., N,1]
        predict = self.projection1(hidden)                                            # [B, T, N, 1]
        return predict


# ---------- Top-level model ----------
class MultiscaleMLP(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.node_size   = model_args["node_size"]
        self.input_len   = model_args["input_len"]
        self.input_dim   = model_args["input_dim"]
        self.output_len  = model_args["output_len"]
        self.patch_len   = model_args["patch_len"]
        self.stride      = model_args["stride"]
        self.td_size     = model_args["td_size"]
        self.dw_size     = model_args["dw_size"]
        self.d_td        = model_args["d_td"]
        self.d_dw        = model_args["d_dw"]
        self.d_d         = model_args["d_d"]
        self.d_spa       = model_args["d_spa"]
        self.if_time_in_day = model_args["if_time_in_day"]
        self.if_day_in_week = model_args["if_day_in_week"]
        self.if_spatial     = model_args["if_spatial"]
        self.num_layer      = model_args["num_layer"]

        # codebooks
        if self.if_time_in_day:
            self.td_codebook = nn.Parameter(torch.empty(self.td_size, self.d_td))
            nn.init.xavier_uniform_(self.td_codebook)
        else:
            self.td_codebook = nn.Parameter(torch.empty(1, 1))
            nn.init.xavier_uniform_(self.td_codebook)

        if self.if_day_in_week:
            self.dw_codebook = nn.Parameter(torch.empty(self.dw_size, self.d_dw))
            nn.init.xavier_uniform_(self.dw_codebook)
        else:
            self.dw_codebook = nn.Parameter(torch.empty(1, 1))
            nn.init.xavier_uniform_(self.dw_codebook)

        if self.if_spatial:
            self.spa_codebook = nn.Parameter(torch.empty(self.node_size, self.d_spa))
            nn.init.xavier_uniform_(self.spa_codebook)
        else:
            self.spa_codebook = nn.Parameter(torch.empty(1, 1))
            nn.init.xavier_uniform_(self.spa_codebook)

        # encoders
        self.patch_encoder = PatchEncoder(
            self.td_size, self.td_codebook, self.dw_codebook, self.spa_codebook,
            self.if_time_in_day, self.if_day_in_week, self.if_spatial,
            self.input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw, self.d_spa,
            self.output_len, self.num_layer
        )

        self.downsamp_encoder = DownsampEncoder(
            self.td_size, self.td_codebook, self.dw_codebook, self.spa_codebook,
            self.if_time_in_day, self.if_day_in_week, self.if_spatial,
            self.input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw, self.d_spa,
            self.output_len, self.num_layer
        )

        future_dim = (
            self.d_td * int(self.if_time_in_day)
            + self.d_dw * int(self.if_day_in_week)
            + self.d_spa * int(self.if_spatial)
        )
        self.future_calendar_head = None
        if future_dim > 0:
            self.future_calendar_head = nn.Sequential(
                nn.Linear(future_dim, self.d_d),
                nn.ReLU(),
                nn.Linear(self.d_d, 1),
            )

        # residual: project last value frame (B,1,N,1) -> (B,T,N,1)
        self.residual = nn.Conv2d(in_channels=1, out_channels=self.output_len, kernel_size=1, bias=True)

    def _future_calendar_term(self, future_data: torch.Tensor, B: int, N: int, device) -> torch.Tensor:
        if future_data is None or self.future_calendar_head is None:
            return torch.zeros(B, self.output_len, N, 1, device=device)
        if future_data.ndim != 4 or future_data.size(1) < self.output_len or future_data.size(2) != N:
            return torch.zeros(B, self.output_len, N, 1, device=device)

        future = future_data[:, : self.output_len]
        features = []
        if self.if_time_in_day:
            tod_index = torch.remainder((future[..., 1] * self.td_size).long(), self.td_size)
            features.append(self.td_codebook[tod_index])
        if self.if_day_in_week:
            dow_index = torch.remainder(future[..., 2].long(), self.dw_size)
            features.append(self.dw_codebook[dow_index])
        if self.if_spatial:
            spatial = self.spa_codebook.view(1, 1, N, self.d_spa).expand(B, self.output_len, N, self.d_spa)
            features.append(spatial)
        if not features:
            return torch.zeros(B, self.output_len, N, 1, device=device)
        return self.future_calendar_head(torch.cat(features, dim=-1))

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None,
                batch_seen: int = 0, epoch: int = 0, train: bool = True, **kwargs) -> torch.Tensor:
        """
        history_data: [B, L, N, C] with C>=3 (value, time_in_day in [0,1), day_in_week as 0..6)
        returns:      [B, T, N]  (last dim squeezed from (B,T,N,1))
        """
        B, L, N, C = history_data.shape
        x = history_data[..., :self.input_dim]   # [B,L,N,C]

        # pad front so length divisible by stride
        in_len_add = ceil(1.0 * self.input_len / self.stride) * self.stride - self.input_len
        if in_len_add > 0:
            pad_block = x[:, -1:, :, :].expand(-1, in_len_add, -1, -1)
            x = torch.cat((pad_block, x), dim=1)

        # downsamp_input: split offsets 0..stride-1 => [B,P,L',N,C] where P==stride
        downsamp_input = [x[:, i::self.stride, :, :] for i in range(self.stride)]
        downsamp_input = torch.stack(downsamp_input, dim=1)

        # patch_input: unfold exact patches of length patch_len => [B,P,L,N,C]
        patch_input = x.unfold(dimension=1, size=self.patch_len, step=self.patch_len).permute(0, 1, 4, 2, 3)

        # encoders
        patch_predict    = self.patch_encoder(patch_input)       # [B,T,N,1]
        downsamp_predict = self.downsamp_encoder(downsamp_input)  # [B,T,N,1]

        # residual on last 'value' frame only
        last_frame = x[:, -1:, :, 0:1].permute(0, 3, 2, 1)  # (B,1,N,1)
        residual_term = self.residual(last_frame)           # (B,T,N,1)
        future_term = self._future_calendar_term(future_data, B, N, x.device)

        out = patch_predict + downsamp_predict + residual_term + future_term  # (B,T,N,1)
        return out.squeeze(-1)  # (B,T,N)
