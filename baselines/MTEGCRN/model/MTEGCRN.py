"""
MTEGCRN: Multi-scale Temporal Enhanced Graph Convolutional Recurrent Network
for Traffic Prediction.

Key components:
1. Temporal Feature Enhancement Module: trainable daily/weekly temporal embeddings
2. Continuous Temporal Learning Module: bidirectional GRU with node-oriented GCN
3. Global Temporal Fusion Module: Transformer for long-range temporal dependencies
4. Multi-scale temporal decomposition: residual decomposition across scales

Reference:
    Yang, S., & Wu, Q. (2025). MTEGCRN: Multi-scale Temporal Enhanced Graph
    Convolutional Recurrent Network for Traffic Prediction. Neurocomputing.
"""
import torch
import torch.nn as nn
import math
from model.NOGCRN import NOGCRN


class GlobalTemporalFusion(nn.Module):
    """Global Temporal Fusion Module using Transformer attention.
    Captures long-range temporal dependencies across the entire sequence.
    """
    def __init__(self, d_model, nhead=4, num_layers=1, dropout=0.1):
        super(GlobalTemporalFusion, self).__init__()
        self.d_model = d_model
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, T, N, D) - temporal sequence of node features
        Returns:
            out: (B, T, N, D) - temporally enhanced features
        """
        B, T, N, D = x.shape
        # Reshape: treat each node independently -> (B*N, T, D)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        out = self.norm(out)
        # Reshape back: (B, N, T, D) -> (B, T, N, D)
        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (B, T, D)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BiCTL(nn.Module):
    """Bidirectional Continuous Temporal Learning module.
    Processes the sequence in both forward and backward directions using
    node-oriented graph convolutional recurrent cells.
    """
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(BiCTL, self).__init__()
        self.node_num = node_num
        self.input_dim = dim_in
        self.dim_out = dim_out

        # Forward and backward recurrent cells
        self.forward_cell = NOGCRN(
            node_num, dim_in, dim_out, cheb_k, embed_dim)
        self.backward_cell = NOGCRN(
            node_num, dim_in, dim_out, cheb_k, embed_dim)

    def forward(self, x, node_embeddings):
        """
        Args:
            x: (B, T, N, input_dim)
            node_embeddings: [dynamic_emb, static_emb]
        Returns:
            h_out: (B, T, N, 2*dim_out) - concatenated bidirectional output
        """
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        B, T, N, _ = x.shape

        # Forward pass
        init_state_f = self.forward_cell.init_hidden_state(B)
        state_f = init_state_f
        forward_states = []
        for t in range(T):
            state_f = self.forward_cell(
                x[:, t, :, :], state_f,
                [node_embeddings[0][:, t, :, :], node_embeddings[1]])
            forward_states.append(state_f)

        # Backward pass
        init_state_b = self.backward_cell.init_hidden_state(B)
        state_b = init_state_b
        backward_states = []
        for t in range(T - 1, -1, -1):
            state_b = self.backward_cell(
                x[:, t, :, :], state_b,
                [node_embeddings[0][:, t, :, :], node_embeddings[1]])
            backward_states.insert(0, state_b)

        # Concatenate forward and backward
        forward_out = torch.stack(forward_states, dim=1)   # (B, T, N, dim_out)
        backward_out = torch.stack(backward_states, dim=1)  # (B, T, N, dim_out)

        h_out = torch.cat([forward_out, backward_out], dim=-1)  # (B, T, N, 2*dim_out)
        return h_out


class MTEGCRN(nn.Module):
    """Multi-scale Temporal Enhanced Graph Convolutional Recurrent Network.

    Architecture:
    1. Temporal Feature Enhancement: modulates node embeddings with
       daily/weekly temporal embeddings
    2. Multi-scale decomposition: extracts traffic patterns at multiple
       temporal scales via residual learning
    3. BiCTL: bidirectional continuous temporal learning at each scale
    4. Global Temporal Fusion: Transformer attention for long-range deps
    5. Prediction: Conv layers to produce multi-step forecasts
    """
    def __init__(self, args):
        super(MTEGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.lag = args.lag
        self.num_layers = args.num_layers
        self.use_D = args.use_day
        self.use_W = args.use_week

        # Dropout layers for each scale
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)

        # --- Temporal Feature Enhancement Module ---
        # Trainable node embeddings (static)
        self.node_embeddings1 = nn.Parameter(
            torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.node_embeddings2 = nn.Parameter(
            torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        # Trainable temporal embeddings for daily and weekly patterns
        self.T_i_D_emb = nn.Parameter(torch.empty(288, args.embed_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.embed_dim))

        # --- Scale 1: Main pattern encoder ---
        self.encoder1 = BiCTL(
            args.num_nodes, args.input_dim, args.rnn_units,
            args.cheb_k, args.embed_dim)

        # --- Scale 2: Residual pattern encoder ---
        self.encoder2 = BiCTL(
            args.num_nodes, args.input_dim, args.rnn_units,
            args.cheb_k, args.embed_dim)

        # --- Scale 3: Fine-grained residual encoder ---
        self.encoder3 = BiCTL(
            args.num_nodes, args.input_dim, args.rnn_units,
            args.cheb_k, args.embed_dim)

        # --- Global Temporal Fusion Module (Transformer) ---
        self.global_fusion = GlobalTemporalFusion(
            d_model=2 * self.hidden_dim,
            nhead=4,
            num_layers=1,
            dropout=0.1)

        # --- Prediction heads ---
        # Scale 1: prediction + residual extraction
        self.end_conv1 = nn.Conv2d(
            1, args.horizon * self.output_dim,
            kernel_size=(1, 2 * self.hidden_dim), bias=True)
        self.end_conv2 = nn.Conv2d(
            1, self.lag * self.output_dim,
            kernel_size=(1, 2 * self.hidden_dim), bias=True)

        # Scale 2: prediction + residual extraction
        self.end_conv3 = nn.Conv2d(
            1, args.horizon * self.output_dim,
            kernel_size=(1, 2 * self.hidden_dim), bias=True)
        self.end_conv4 = nn.Conv2d(
            1, self.lag * self.output_dim,
            kernel_size=(1, 2 * self.hidden_dim), bias=True)

        # Scale 3: prediction
        self.end_conv5 = nn.Conv2d(
            1, args.horizon * self.output_dim,
            kernel_size=(1, 2 * self.hidden_dim), bias=True)

    def forward(self, source, i=2):
        """
        Args:
            source: (B, T, N, F) where F includes [traffic, time_of_day, day_of_week]
            i: mode flag (1=single scale, 2=multi-scale)
        Returns:
            output: (B, horizon, N, output_dim) predictions
        """
        # --- Temporal Feature Enhancement ---
        node_embedding1 = self.node_embeddings1  # (N, embed_dim)

        if self.use_D:
            t_i_d_data = source[..., 1]  # (B, T, N)
            T_i_D_emb = self.T_i_D_emb[
                (t_i_d_data * 288).type(torch.LongTensor)]  # (B, T, N, embed_dim)
            node_embedding1 = torch.mul(node_embedding1, T_i_D_emb)

        if self.use_W:
            d_i_w_data = source[..., 2]  # (B, T, N)
            D_i_W_emb = self.D_i_W_emb[
                (d_i_w_data).type(torch.LongTensor)]  # (B, T, N, embed_dim)
            node_embedding1 = torch.mul(node_embedding1, D_i_W_emb)

        node_embeddings = [node_embedding1, self.node_embeddings1]

        # Extract traffic feature only
        source = source[..., 0].unsqueeze(-1)  # (B, T, N, 1)

        if i == 1:
            # Single-scale mode (for potential pre-training)
            output = self.encoder1(source, node_embeddings)
            output = self.dropout1(output[:, -1:, :, :])
            output1 = self.end_conv1(output)
            return output1
        else:
            # --- Multi-scale temporal decomposition ---

            # Scale 1: main temporal pattern
            output = self.encoder1(source, node_embeddings)  # (B, T, N, 2*H)

            # Apply Global Temporal Fusion (Transformer)
            output = self.global_fusion(output)  # (B, T, N, 2*H)

            output = self.dropout1(output[:, -1:, :, :])  # (B, 1, N, 2*H)

            # Scale 1 predictions
            source1 = self.end_conv2(output)  # reconstruct for residual
            output = self.end_conv1(output)    # prediction

            # Compute residual: original - reconstructed
            source1 = source - source1  # (B, T, N, 1)

            # Scale 2: residual pattern
            output1 = self.encoder2(source1, node_embeddings)  # (B, T, N, 2*H)
            output1 = self.dropout2(output1[:, -1:, :, :])

            source2 = self.end_conv4(output1)  # reconstruct for residual
            output1 = self.end_conv3(output1)   # prediction

            # Scale 3: fine-grained residual
            source3 = source1 - source2  # (B, T, N, 1)
            output2 = self.encoder3(source3, node_embeddings)  # (B, T, N, 2*H)
            output2 = self.dropout3(output2[:, -1:, :, :])
            output2 = self.end_conv5(output2)  # prediction

            # Aggregate multi-scale predictions
            return output + output1 + output2
