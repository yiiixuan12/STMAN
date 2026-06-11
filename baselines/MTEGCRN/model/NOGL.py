"""
Node-Oriented Graph Learning (NOGL) module for MTEGCRN.
Breaks the shared parameter paradigm by allocating personalized parameter
spaces to each node, enabling capture of node-specific traffic patterns.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict


class NOGL(nn.Module):
    """Node-Oriented Graph Learning layer.
    Similar to DEGL in DMFGCRN but with enhanced node-specific parameterization.
    """
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(NOGL, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, dim_out))
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.embed_dim = embed_dim
        self.fc = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                ('sigmoid1', nn.Sigmoid()),
                ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                ('sigmoid2', nn.Sigmoid()),
                ('fc3', nn.Linear(self.middle_dim, self.embed_dim))
            ]))

    def forward(self, x, node_embeddings):
        """
        Args:
            x: (B, N, dim_in)
            node_embeddings: list of [dynamic_emb, static_emb]
                dynamic_emb: (B, N, embed_dim) - temporally modulated
                static_emb: (N, embed_dim) - base node embeddings
        """
        node_num = node_embeddings[0].shape[1] if isinstance(
            node_embeddings, list) else x.shape[1]
        supports1 = torch.eye(node_num).to(x.device)

        # Dynamic graph learning via hyper-network
        filter_val = self.fc(x)
        nodevec = torch.tanh(torch.mul(node_embeddings[0], filter_val))
        supports2 = self.get_laplacian(
            F.relu(torch.matmul(nodevec, nodevec.transpose(2, 1))),
            supports1)

        # Node-specific weights from static embeddings
        weights = torch.einsum('nd,dkio->nkio',
                               node_embeddings[1], self.weights_pool)
        bias = torch.matmul(node_embeddings[1], self.bias_pool)

        # Graph convolution with Chebyshev polynomials
        x_g1 = torch.einsum("nm,bmc->bnc", supports1, x)
        if len(supports2.shape) == 3:
            x_g2 = torch.einsum("bnm,bmc->bnc", supports2, x)
        else:
            x_g2 = torch.einsum("nm,bmc->bnc", supports2, x)
        x_g = torch.stack([x_g1, x_g2], dim=1)

        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias

        return x_gconv

    @staticmethod
    def get_laplacian(graph, I, normalize=True):
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        else:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        return L
