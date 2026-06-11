import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class DEGL(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(DEGL, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k,dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.embed_dim = embed_dim
        self.fc=nn.Sequential(
                OrderedDict([('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(self.middle_dim, self.embed_dim))]))
 
        import os
        graph_type = os.environ.get('GRAPH_TYPE', 'dynamic')  
        
        self.graph_type = graph_type
        self.predefined_adj = None

        if graph_type != 'dynamic' and graph_type != 'static_adaptive':
            if graph_type == 'distance':
                adj_path = 'data/adj_distance_04.npy'  
                print(f"Using distance-based adjacency matrix from: {adj_path}")
            elif graph_type == 'connectivity':
                adj_path = 'data/adj_selfloop_PEMS04.npy' 
                print(f"Using connectivity-based adjacency matrix from: {adj_path}")
            elif graph_type == 'dtw':
                adj_path = 'data/dtw_adj01_PEMS04.npy'  
                print(f"Using DTW-based adjacency matrix from: {adj_path}")
            else:
                raise ValueError(f"Unknown graph type: {graph_type}")
            
            adj_matrix = np.load(adj_path)
            self.predefined_adj = torch.FloatTensor(adj_matrix)
            
            num_nodes = adj_matrix.shape[0]
            self.node_embeddings_fixed = nn.Parameter(torch.randn(num_nodes, embed_dim))
            
        elif graph_type == 'static_adaptive':
            self.node_embeddings_static = nn.Parameter(torch.randn(embed_dim, embed_dim))
        
    def forward(self, x, node_embeddings):
        node_num = node_embeddings[0].shape[1] if isinstance(node_embeddings, list) else x.shape[1]
        supports1 = torch.eye(node_num).to(x.device)
        
        if self.graph_type in ['distance', 'connectivity', 'dtw']:
            self.predefined_adj = self.predefined_adj.to(x.device)
            supports2 = self.get_laplacian(self.predefined_adj, supports1)
            weights = torch.einsum('nd,dkio->nkio', self.node_embeddings_fixed, self.weights_pool)
            bias = torch.matmul(self.node_embeddings_fixed, self.bias_pool)
            
        elif self.graph_type == 'static_adaptive':
            static_adj = F.softmax(
                F.relu(torch.mm(self.node_embeddings_static, self.node_embeddings_static.transpose(0, 1))), 
                dim=1
            )
            supports2 = self.get_laplacian(static_adj, supports1)
            weights = torch.einsum('nd,dkio->nkio', self.node_embeddings_static, self.weights_pool)
            bias = torch.matmul(self.node_embeddings_static, self.bias_pool)
            
        else:
            filter = self.fc(x)
            nodevec = torch.tanh(torch.mul(node_embeddings[0], filter))
            supports2 = self.get_laplacian(F.relu(torch.matmul(nodevec, nodevec.transpose(2, 1))), supports1)
            weights = torch.einsum('nd,dkio->nkio', node_embeddings[1], self.weights_pool)
            bias = torch.matmul(node_embeddings[1], self.bias_pool)
        
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
