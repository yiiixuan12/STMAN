# utils.py
# ----------------------------
# 数据预处理、数据标准化、读取数据
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
    header=None,        # 若CSV无表头，设None；有表头就设0或"infer"
    symmetrize=True,    # 是否对称化 A <- (A + A^T)/2
    clip_negative=True, # 去除负值（有些CSV可能有-0等）
    add_self_loops=True,# 是否加自环
    self_loop_weight=1.0,
    normalize=None,     # None / "row" / "sym"
    dtype=torch.float32,
    device="cpu"
):
    # 1) 读取
    A = pd.read_csv(path, header=header).values.astype(np.float32)

    # 2) 形状检查
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape {A.shape}")
    if n_nodes is not None and A.shape[0] != n_nodes:
        raise ValueError(f"Adjacency size {A.shape[0]} != n_nodes {n_nodes}")

    # 3) 清洗
    if clip_negative:
        A = np.clip(A, a_min=0.0, a_max=None)
    if symmetrize:
        A = 0.5 * (A + A.T)

    # 4) 加自环（可选，提升稳定性）
    if add_self_loops:
        A = A.copy()
        np.fill_diagonal(A, A.diagonal() + self_loop_weight)

    # 5) 归一化（可选）
    if normalize == "row":
        # D^{-1} A
        row_sum = A.sum(axis=1, keepdims=True) + 1e-12
        A = A / row_sum
    elif normalize == "sym":
        # D^{-1/2} A D^{-1/2}
        d = A.sum(axis=1) + 1e-12
        d_inv_sqrt = 1.0 / np.sqrt(d)
        A = (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]

    # 6) 转 tensor
    A_t = torch.tensor(A, dtype=dtype, device=device)
    return A_t

# ====== 新增：为 STGCN 构造 GSO（图移位算子） ======
def build_gso_for_stgcn(
    adj: torch.Tensor,
    graph_conv_type: str = "cheb_graph_conv",
    add_self_loops: bool = True,
) -> torch.Tensor:
    """
    根据原始邻接 adj 构造 STGCN 需要的 GSO：
      - graph_conv_type == 'graph_conv':  D^{-1/2} (A + I) D^{-1/2}
      - graph_conv_type == 'cheb_graph_conv':  ˜L = 2L/λ_max - I, 其中 L = I - D^{-1/2} A D^{-1/2}, λ_max≈2
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
        # 直接把 Ā 作为 GSO
        return A_sym
    elif graph_conv_type == "cheb_graph_conv":
        # 归一化拉普拉斯 L，并缩放到[-1,1]
        L = I - A_sym
        lambda_max = 2.0
        return 2.0 * L / lambda_max - I
    else:
        raise ValueError(f"Unknown graph_conv_type: {graph_conv_type}")

# ====== 标准化器 ======
class StandardScaler:
    """
    标准化器，用于数据的归一化和反归一化
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
            raise ValueError("必须先调用fit方法")
        return (data - self.mean) / self.std
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("必须先调用fit方法")
        return data * self.std + self.mean

class MinMaxScaler:
    """
    最小-最大标准化器，将数据缩放到[0,1]范围
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
            raise ValueError("必须先调用fit方法")
        return (data - self.min_val) / self.scale
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        if self.min_val is None or self.scale is None:
            raise ValueError("必须先调用fit方法")
        return data * self.scale + self.min_val

# ====== 数据预处理/Loader ======
def preprocess_data(X_train, X_val, X_test, scaler_type='standard', verbose=True):
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type 必须是 'standard' 或 'minmax'")
    
    if verbose:
        print("=" * 60)
        print(f"训练集形状: {X_train.shape}")

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    if verbose:
        print(f"训练集 - 均值: {np.mean(X_train_scaled):.4f}, 标准差: {np.std(X_train_scaled):.4f}")
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
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
        print(f"测试批次数: {len(test_loader)}")
        print(f"批大小: {batch_size}")
    
    return train_loader, val_loader, test_loader, scaler
