#utils.py
# ----------------------------
# 数据预处理、数据标准化、读取数据
# ----------------------------
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import math
from torch.optim.lr_scheduler import _LRScheduler


def append_calendar_features(raw, steps_per_day=288, add_dow=False, start_index=0):
    """Append raw TOD and optional DOW channels after target channels."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 2:
        raw = raw[:, :, np.newaxis]
    if raw.ndim != 3:
        raise ValueError(f"raw must be [T,N,F], got shape {raw.shape}")
    total, n_nodes = raw.shape[0], raw.shape[1]
    steps = np.arange(start_index, start_index + total, dtype=np.int64)
    tod = ((steps % steps_per_day) / float(steps_per_day)).astype(np.float32)
    features = [np.broadcast_to(tod[:, None, None], (total, n_nodes, 1)).copy()]
    if add_dow:
        dow = (((steps // steps_per_day) % 7) / 7.0).astype(np.float32)
        features.append(np.broadcast_to(dow[:, None, None], (total, n_nodes, 1)).copy())
    return np.concatenate([raw] + features, axis=-1)

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

# 数据标准化类
class StandardScaler:
    """
    标准化器，用于数据的归一化和反归一化
    """
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        """
        计算训练数据的均值和标准差
        Args:
            data: 训练数据 [T, N, F] 或 [T, N] 或 [samples, features]
        """
        data = np.array(data)
        # 计算所有维度的均值和标准差
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        # 避免除零错误
        self.std = np.where(self.std == 0, 1.0, self.std)
        
    def transform(self, data):
        """
        标准化数据
        """
        if self.mean is None or self.std is None:
            raise ValueError("必须先调用fit方法")
        return (data - self.mean) / self.std
    
    def fit_transform(self, data):
        """
        拟合并转换数据
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        """
        反标准化数据（用于恢复预测结果）
        """
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
        """
        计算训练数据的最小值和最大值
        """
        data = np.array(data)
        self.min_val = np.min(data, axis=0, keepdims=True)
        self.max_val = np.max(data, axis=0, keepdims=True)
        self.scale = self.max_val - self.min_val
        # 避免除零错误
        self.scale = np.where(self.scale == 0, 1.0, self.scale)
    
    def transform(self, data):
        """
        最小-最大标准化
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("必须先调用fit方法")
        return (data - self.min_val) / self.scale
    
    def fit_transform(self, data):
        """
        拟合并转换数据
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        """
        反标准化数据
        """
        if self.min_val is None or self.scale is None:
            raise ValueError("必须先调用fit方法")
        return data * self.scale + self.min_val

# 数据预处理函数
def preprocess_data(X_train, X_val, X_test, scaler_type='standard', verbose=True,
                    scale_target_only=False, target_dim=1):
    """
    数据预处理和归一化
    
    Args:
        X_train, X_val, X_test: 原始数据 [T, N, F]
        scaler_type: 'standard' 或 'minmax'
        verbose: 是否打印统计信息
    
    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled: 归一化后的数据
        scaler: 标准化器（用于反标准化）
    """
    # 选择标准化器
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type 必须是 'standard' 或 'minmax'")
    
    if verbose:
        print("=" * 60)
        # print("数据预处理统计:")
        print(f"训练集形状: {X_train.shape}")
        # print(f"验证集形状: {X_val.shape}")
        # print(f"测试集形状: {X_test.shape}")
        # print(f"标准化方法: {scaler_type}")
        
        # print("\n原始数据统计:")
        # print(f"训练集 - 均值: {np.mean(X_train):.4f}, 标准差: {np.std(X_train):.4f}")
        # print(f"训练集 - 最小值: {np.min(X_train):.4f}, 最大值: {np.max(X_train):.4f}")
    
    # 只用训练集拟合标准化器
    scaler.fit(X_train)
    if scale_target_only and X_train.shape[-1] > target_dim:
        if hasattr(scaler, "mean"):
            scaler.mean = scaler.mean.copy()
            scaler.std = scaler.std.copy()
            scaler.mean[..., target_dim:] = 0.0
            scaler.std[..., target_dim:] = 1.0
        elif hasattr(scaler, "min_val"):
            scaler.min_val = scaler.min_val.copy()
            scaler.scale = scaler.scale.copy()
            scaler.min_val[..., target_dim:] = 0.0
            scaler.scale[..., target_dim:] = 1.0
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    if verbose:
        # print(f"\n标准化后数据统计:")
        print(f"训练集 - 均值: {np.mean(X_train_scaled):.4f}, 标准差: {np.std(X_train_scaled):.4f}")
        # print(f"训练集 - 最小值: {np.min(X_train_scaled):.4f}, 最大值: {np.max(X_train_scaled):.4f}")
        print("=" * 60)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


class TrafficDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
            self.data = torch.as_tensor(data, dtype=torch.float32)
            self.seq_len = seq_len
            self.pred_len = pred_len
            
    def __len__(self):
            return len(self.data) - self.seq_len - self.pred_len + 1
            
    def __getitem__(self, idx):
            x = self.data[idx:idx + self.seq_len]
            y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
            return x, y


def _build_loader(dataset, batch_size, shuffle, num_workers):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


# 数据加载器创建函数
def create_data_loaders(X_train, X_val, X_test, seq_len=12, pred_len=12, batch_size=32, 
                       scaler_type='standard', verbose=True,
                       train_workers=4, eval_workers=2,
                       scale_target_only=False, target_dim=1):
    """
    创建数据加载器（包含数据预处理）
    
    Args:
        X_train, X_val, X_test: 训练、验证、测试数据 [T, N, F]
        seq_len: 输入序列长度
        pred_len: 预测序列长度
        batch_size: 批大小
        scaler_type: 标准化类型 'standard' 或 'minmax'
        verbose: 是否打印详细信息
    """
    
    # 数据预处理和归一化
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocess_data(
        X_train, X_val, X_test, scaler_type=scaler_type, verbose=verbose,
        scale_target_only=scale_target_only, target_dim=target_dim
    )
    
    # 创建数据集
    train_dataset = TrafficDataset(X_train_scaled, seq_len, pred_len)
    val_dataset = TrafficDataset(X_val_scaled, seq_len, pred_len)
    test_dataset = TrafficDataset(X_test_scaled, seq_len, pred_len)
    
    # 创建数据加载器
    train_loader = _build_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=train_workers)
    val_loader = _build_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=eval_workers)
    test_loader = _build_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=eval_workers)
    
    if verbose:
        # print(f"\n数据加载器信息:")
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
        print(f"测试批次数: {len(test_loader)}")
        print(f"批大小: {batch_size}")
    
    return train_loader, val_loader, test_loader, scaler

