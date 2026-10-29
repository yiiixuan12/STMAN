#utils.py
# ----------------------------
# Data preprocessing, scaling, and loading
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
    header=None,        # Use None for a CSV without a header; otherwise use 0 or "infer"
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
    
    # 2) Validate shape
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

    # 6) Convert to tensor
    A_t = torch.tensor(A, dtype=dtype, device=device)
    return A_t

# Data scaler classes
class StandardScaler:
    """
    Scaler for normalizing and inverse-transforming data
    """
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        """
        Compute the mean and standard deviation of the training data
        Args:
            data: Training data [T, N, F], [T, N], or [samples, features]
        """
        data = np.array(data)
        # Compute the mean and standard deviation across all dimensions
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        # Avoid division by zero
        self.std = np.where(self.std == 0, 1.0, self.std)
        
    def transform(self, data):
        """
        Standardize data
        """
        if self.mean is None or self.std is None:
            raise ValueError("Call fit before using this method")
        return (data - self.mean) / self.std
    
    def fit_transform(self, data):
        """
        Fit and transform data
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        """
        Inverse-transform data to recover predictions on the original scale
        """
        if self.mean is None or self.std is None:
            raise ValueError("Call fit before using this method")
        return data * self.std + self.mean

class MinMaxScaler:
    """
    Min-max scaler that scales data to the [0,1] range
    """
    def __init__(self):
        self.min_val = None
        self.max_val = None
        self.scale = None
    
    def fit(self, data):
        """
        Compute the minimum and maximum of the training data
        """
        data = np.array(data)
        self.min_val = np.min(data, axis=0, keepdims=True)
        self.max_val = np.max(data, axis=0, keepdims=True)
        self.scale = self.max_val - self.min_val
        # Avoid division by zero
        self.scale = np.where(self.scale == 0, 1.0, self.scale)
    
    def transform(self, data):
        """
        Min-max scaling
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("Call fit before using this method")
        return (data - self.min_val) / self.scale
    
    def fit_transform(self, data):
        """
        Fit and transform data
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        """
        Inverse-transform data
        """
        if self.min_val is None or self.scale is None:
            raise ValueError("Call fit before using this method")
        return data * self.scale + self.min_val

# Data preprocessing functions
def preprocess_data(X_train, X_val, X_test, scaler_type='standard', verbose=True,
                    scale_target_only=False, target_dim=1):
    """
    Data preprocessing and normalization
    
    Args:
        X_train, X_val, X_test: Raw data [T, N, F]
        scaler_type: 'standard' or 'minmax'
        verbose: Whether to print statistics
    
    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled: Normalized data
        scaler: Scaler for inverse transformation
    """
    # Select the scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    if verbose:
        print("=" * 60)
        # print("Data preprocessing statistics:")
        print(f"Training set shape: {X_train.shape}")
        # print(f"Validation set shape: {X_val.shape}")
        # print(f"Test set shape: {X_test.shape}")
        # print(f"Scaling method: {scaler_type}")
        
        # print("\nRaw data statistics:")
        # print(f"Training set - mean: {np.mean(X_train):.4f}, standard deviation: {np.std(X_train):.4f}")
        # print(f"Training set - minimum: {np.min(X_train):.4f}, maximum: {np.max(X_train):.4f}")
    
    # Fit the scaler using only the training set
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
        # print(f"\nScaled data statistics:")
        print(f"Training set - mean: {np.mean(X_train_scaled):.4f}, standard deviation: {np.std(X_train_scaled):.4f}")
        # print(f"Training set - minimum: {np.min(X_train_scaled):.4f}, maximum: {np.max(X_train_scaled):.4f}")
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


# Data loader creation functions
def create_data_loaders(X_train, X_val, X_test, seq_len=12, pred_len=12, batch_size=32, 
                       scaler_type='standard', verbose=True,
                       train_workers=4, eval_workers=2,
                       scale_target_only=False, target_dim=1):
    """
    Create data loaders, including data preprocessing
    
    Args:
        X_train, X_val, X_test: Training, validation, and test data [T, N, F]
        seq_len: Input sequence length
        pred_len: Prediction sequence length
        batch_size: Batch size
        scaler_type: Scaling type, 'standard' or 'minmax'
        verbose: Whether to print details
    """
    
    # Data preprocessing and normalization
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocess_data(
        X_train, X_val, X_test, scaler_type=scaler_type, verbose=verbose,
        scale_target_only=scale_target_only, target_dim=target_dim
    )
    
    # Create datasets
    train_dataset = TrafficDataset(X_train_scaled, seq_len, pred_len)
    val_dataset = TrafficDataset(X_val_scaled, seq_len, pred_len)
    test_dataset = TrafficDataset(X_test_scaled, seq_len, pred_len)
    
    # Create data loaders
    train_loader = _build_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=train_workers)
    val_loader = _build_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=eval_workers)
    test_loader = _build_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=eval_workers)
    
    if verbose:
        # print(f"\nData loader information:")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader, scaler

