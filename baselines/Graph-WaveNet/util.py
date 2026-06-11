import pickle
import numpy as np
import os
import torch
try:
    import scipy.sparse as sp
    from scipy.sparse import linalg
except ModuleNotFoundError:
    sp = None
    linalg = None


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StreamingWindowDataLoader(object):
    def __init__(self, full, positions, x_offsets, y_offsets, batch_size, scaler):
        self.full = full
        self.positions = np.asarray(positions, dtype=np.int64)
        self.x_offsets = np.asarray(x_offsets, dtype=np.int64)
        self.y_offsets = np.asarray(y_offsets, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.scaler = scaler
        self.current_ind = 0
        self.size = int(len(self.positions))
        self.num_batch = int(np.ceil(self.size / self.batch_size)) if self.size else 0

    def shuffle(self):
        self.positions = self.positions[np.random.permutation(self.size)]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                batch_pos = self.positions[start_ind:end_ind]
                x_i = self.full[batch_pos[:, None] + self.x_offsets[None, :]].copy()
                y_i = self.full[batch_pos[:, None] + self.y_offsets[None, :]].copy()
                x_i[..., 0] = self.scaler.transform(x_i[..., 0])
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    if sp is None:
        adj = np.asarray(adj, dtype=np.float32)
        rowsum = adj.sum(axis=1)
        d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
        d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
        return (d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]).astype(np.float32)
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    if sp is None:
        adj = np.asarray(adj, dtype=np.float32)
        rowsum = adj.sum(axis=1)
        d_inv = np.divide(1.0, rowsum, out=np.zeros_like(rowsum, dtype=np.float32), where=rowsum != 0)
        return (d_inv[:, None] * adj).astype(np.float32)
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    if sp is None:
        adj = np.asarray(adj, dtype=np.float32)
        rowsum = adj.sum(axis=1)
        d_inv_sqrt = np.divide(1.0, np.sqrt(rowsum), out=np.zeros_like(rowsum, dtype=np.float32), where=rowsum != 0)
        normalized_adj = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
        return (np.eye(adj.shape[0], dtype=np.float32) - normalized_adj).astype(np.float32)
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if sp is None:
        if lambda_max is None:
            lambda_max = np.linalg.eigvals(L).real.max()
        M = L.shape[0]
        return ((2 / lambda_max * L) - np.eye(M, dtype=np.float32)).astype(np.float32)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


def _load_raw_stream_source(source_path):
    source_path = os.fspath(source_path)
    if source_path.endswith(".npz"):
        raw = np.load(source_path)["data"]
    elif source_path.endswith(".h5") or source_path.endswith(".hdf5"):
        import pandas as pd
        try:
            raw = pd.read_hdf(source_path).values
        except ImportError:
            directory = os.path.dirname(source_path)
            stem_csv = os.path.splitext(source_path)[0] + ".csv"
            candidates = [stem_csv]
            candidates.extend(
                os.path.join(directory, name)
                for name in os.listdir(directory)
                if name.lower().endswith(".csv")
                and "adj" not in name.lower()
                and "edge" not in name.lower()
                and "location" not in name.lower()
            )
            for candidate in candidates:
                if os.path.exists(candidate):
                    raw = pd.read_csv(candidate, header=None).values
                    break
            else:
                raise
    else:
        raw = np.loadtxt(source_path, delimiter=",")
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 2:
        raw = raw[:, :, None]
    if raw.ndim != 3:
        raise ValueError(f"Expected raw series with shape [T, N] or [T, N, C], got {raw.shape}")
    target = raw[..., :1]
    steps = np.arange(target.shape[0], dtype=np.float32)
    tod = ((steps % 288) / 288.0).reshape(-1, 1, 1)
    tod = np.broadcast_to(tod, target.shape).astype(np.float32)
    return np.concatenate([target, tod], axis=-1).astype(np.float32)


def _default_stream_split(source_path):
    key = os.path.basename(str(source_path)).upper().replace("-", "").replace("_", "")
    if "METRLA" in key or "PEMSBAY" in key:
        return 0.7, 0.1, 0.2
    return 0.6, 0.2, 0.2


def load_dataset_streaming(
    source_path,
    seq_length_x,
    horizon,
    batch_size,
    valid_batch_size=None,
    test_batch_size=None,
    train_ratio=None,
    val_ratio=None,
    test_ratio=None,
    y_start=1,
):
    full = _load_raw_stream_source(source_path)
    default_train, default_val, default_test = _default_stream_split(source_path)
    train_ratio = default_train if train_ratio is None else train_ratio
    val_ratio = default_val if val_ratio is None else val_ratio
    test_ratio = default_test if test_ratio is None else test_ratio
    x_offsets = np.sort(np.arange(-(int(seq_length_x) - 1), 1, 1))
    y_offsets = np.sort(np.arange(int(y_start), int(horizon) + 1, 1))
    min_t = abs(min(x_offsets))
    max_t = full.shape[0] - abs(max(y_offsets))
    if max_t <= min_t:
        raise ValueError(
            f"Not enough timesteps for seq_length_x={seq_length_x} horizon={horizon}: T={full.shape[0]}"
        )
    positions = np.arange(min_t, max_t, dtype=np.int64)

    total = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    num_samples = int(len(positions))
    num_train = round(num_samples * float(train_ratio))
    num_test = round(num_samples * float(test_ratio))
    num_val = num_samples - num_train - num_test
    train_pos = positions[:num_train]
    val_pos = positions[num_train:num_train + num_val]
    test_pos = positions[-num_test:]

    train_end = int(train_pos[-1] + 1) if len(train_pos) else int(min_t + 1)
    train_target = full[:train_end, :, 0]
    scaler = StandardScaler(mean=train_target.mean(), std=max(float(train_target.std()), 1e-6))

    data = {
        "train_loader": StreamingWindowDataLoader(
            full, train_pos, x_offsets, y_offsets, batch_size, scaler
        ),
        "val_loader": StreamingWindowDataLoader(
            full, val_pos, x_offsets, y_offsets, valid_batch_size or batch_size, scaler
        ),
        "test_loader": StreamingWindowDataLoader(
            full, test_pos, x_offsets, y_offsets, test_batch_size or batch_size, scaler
        ),
        "scaler": scaler,
    }
    return data


def load_dataset_eval_only(dataset_dir, test_batch_size):
    data = {}
    train_data = np.load(os.path.join(dataset_dir, 'train.npz'))
    x_train = train_data['x']
    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())
    del x_train
    test_data = np.load(os.path.join(dataset_dir, 'test.npz'))
    data['x_test'] = test_data['x']
    data['y_test'] = test_data['y']
    data['x_test'][..., 0] = scaler.transform(data['x_test'][..., 0])
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse
