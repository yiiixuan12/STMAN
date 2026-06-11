"""
Node-Oriented Graph Convolutional Recurrent Network (NOGCRN) cell.
GRU-based recurrent cell with node-oriented graph convolution,
forming the continuous temporal learning module.
"""
import torch
import torch.nn as nn
from model.NOGL import NOGL


class NOGCRN(nn.Module):
    """GRU cell with Node-Oriented Graph Learning."""
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(NOGCRN, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = NOGL(dim_in + self.hidden_dim, 2 * dim_out,
                         cheb_k, embed_dim)
        self.update = NOGL(dim_in + self.hidden_dim, dim_out,
                           cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        """
        Args:
            x: (B, N, input_dim)
            state: (B, N, hidden_dim)
            node_embeddings: [dynamic_emb, static_emb]
        Returns:
            h: (B, N, hidden_dim) - new hidden state
        """
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
