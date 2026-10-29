# utils.py
# ----------------------------
# Data preprocessing, scaling, and loading
# ----------------------------
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import math
from torch.optim.lr_scheduler import _LRScheduler

def load_adjacency_csv(
    path,
    n_nodes=None,
    header=None,        # Use None for a headerless CSV, or 0/"infer" for a header
    symmetrize=True,    # Whether to symmetrize A <- (A + A^T)/2
    clip_negative=True, # Remove negative values (some CSVs may contain -0, etc.)
    add_self_loops=True,# Whether to add self-loops
    self_loop_weight=1.0,
    normalize=None,     # None / "row" / "sym"
    dtype=torch.float32,
    device="cpu"
):
    # 1) Read
    A = pd.read_csv(path, header=header).values.astype(np.float32)

    # 2) Check shape
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape {A.shape}")
    if n_nodes is not None and A.shape[0] != n_nodes:
        raise ValueError(f"Adjacency size {A.shape[0]} != n_nodes {n_nodes}")

    # 3) Clean
    if clip_negative:
        A = np.clip(A, a_min=0.0, a_max=None)
    if symmetrize:
        A = 0.5 * (A + A.T)

    # 4) Add self-loops (optional, improves stability)
    if add_self_loops:
        A = A.copy()
        np.fill_diagonal(A, A.diagonal() + self_loop_weight)

    # 5) Normalize (optional)
    if normalize == "row":
        # D^{-1} A
        row_sum = A.sum(axis=1, keepdims=True) + 1e-12
        A = A / row_sum
    elif normalize == "sym":
        # D^{-1/2} A D^{-1/2}
        d = A.sum(axis=1) + 1e-12
        d_inv_sqrt = 1.0 / np.sqrt(d)
        A = (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]

    # 6) Convert to a tensor
    A_t = torch.tensor(A, dtype=dtype, device=device)
    return A_t

# ====== Build the GSO (graph shift operator) for STGCN ======
def build_gso_for_stgcn(
    adj: torch.Tensor,
    graph_conv_type: str = "cheb_graph_conv",
    add_self_loops: bool = True,
) -> torch.Tensor:
    """
    Build the GSO required by STGCN from the raw adjacency adj:
      - graph_conv_type == 'graph_conv':  D^{-1/2} (A + I) D^{-1/2}
      - graph_conv_type == 'cheb_graph_conv':  ˜L = 2L/λ_max - I, where L = I - D^{-1/2} A D^{-1/2}, λ_max≈2
    """
    if adj.ndim != 2 or adj.size(0) != adj.size(1):
        raise ValueError(f"adj must be square, got {tuple(adj.shape)}")

    A = adj.clone()
    A = torch.clamp(A, min=0)
    N = A.size(0)
    I = torch.eye(N, dtype=A.dtype, device=A.device)

    if add_self_loops:
        A = A + I

    deg = torch.clamp(A.sum(dim=1), min=1e-12)
    D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    A_sym = D_inv_sqrt @ A @ D_inv_sqrt  # D^{-1/2} A D^{-1/2}

    if graph_conv_type == "graph_conv":
        # Use Ā directly as the GSO
        return A_sym
    elif graph_conv_type == "cheb_graph_conv":
        # Normalize the Laplacian L and scale it to [-1,1]
        L = I - A_sym
        lambda_max = 2.0
        return 2.0 * L / lambda_max - I
    else:
        raise ValueError(f"Unknown graph_conv_type: {graph_conv_type}")

# ====== Scalers ======
class StandardScaler:
    """
    Standard scaler for normalization and inverse transformation
    """
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        data = np.array(data)
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std = np.where(self.std == 0, 1.0, self.std)
        
    def transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("Call fit before using this method")
        return (data - self.mean) / self.std
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("Call fit before using this method")
        return data * self.std + self.mean

class MinMaxScaler:
    """
    Min-max scaler that scales data to [0,1]
    """
    def __init__(self):
        self.min_val = None
        self.max_val = None
        self.scale = None
    
    def fit(self, data):
        data = np.array(data)
        self.min_val = np.min(data, axis=0, keepdims=True)
        self.max_val = np.max(data, axis=0, keepdims=True)
        self.scale = self.max_val - self.min_val
        self.scale = np.where(self.scale == 0, 1.0, self.scale)
    
    def transform(self, data):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Call fit before using this method")
        return (data - self.min_val) / self.scale
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        if self.min_val is None or self.scale is None:
            raise ValueError("Call fit before using this method")
        return data * self.scale + self.min_val

# ====== Data preprocessing/loaders ======
def preprocess_data(X_train, X_val, X_test, scaler_type='standard', verbose=True):
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    if verbose:
        print("=" * 60)
        print(f"Training set shape: {X_train.shape}")

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    if verbose:
        print(f"Training set - mean: {np.mean(X_train_scaled):.4f}, standard deviation: {np.std(X_train_scaled):.4f}")
        print("=" * 60)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


class TrafficDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
            
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
            
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def create_data_loaders(X_train, X_val, X_test, seq_len=12, pred_len=12, batch_size=32, 
                       scaler_type='standard', verbose=True):
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocess_data(
        X_train, X_val, X_test, scaler_type=scaler_type, verbose=verbose
    )
    
    train_dataset = TrafficDataset(X_train_scaled, seq_len, pred_len)
    val_dataset = TrafficDataset(X_val_scaled, seq_len, pred_len)
    test_dataset = TrafficDataset(X_test_scaled, seq_len, pred_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    if verbose:
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader, scaler
