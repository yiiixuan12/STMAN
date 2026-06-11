import os
import numpy as np


def load_st_dataset(dataset):
    """Load spatio-temporal dataset.
    Output shape: (T, N, 1) - only traffic flow/speed feature.
    """
    custom_root = os.environ.get("TABLE4_CUSTOM_DATA_ROOT")
    if custom_root:
        aliases = {
            "METRLA": ("METR-LA", "METR-LA"),
            "PEMSBAY": ("PEMS-BAY", "PEMS-BAY"),
            "PEMSD3": ("PEMS03", "PEMS03"),
            "PEMSD7": ("PEMS07", "PEMS07"),
            "PEMSD8": ("PEMS08", "PEMS08"),
        }
        candidates = [(dataset, dataset)]
        if dataset in aliases:
            candidates.insert(0, aliases[dataset])
        data = None
        for folder, stem in candidates:
            custom_path = os.path.join(custom_root, folder, f"{stem}.npz")
            if os.path.exists(custom_path):
                data = np.load(custom_path)['data'][:, :, 0]
                break
    else:
        data = None

    if data is not None:
        pass
    elif dataset == 'PEMSD3':
        data_path = os.path.join('./data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  # traffic flow
    elif dataset == 'PEMSD8':
        data_path = os.path.join('./data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  # traffic flow
    elif dataset == 'PEMSD4':
        data_path = os.path.join('./data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  # traffic flow
    elif dataset == 'PEMSD7':
        data_path = os.path.join('./data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  # traffic flow
    elif dataset == 'METRLA':
        data_path = os.path.join('./data/METRLA/METRLA.npz')
        data = np.load(data_path)['data'][:, :, 0]  # traffic speed
    elif dataset == 'PEMSBAY':
        data_path = os.path.join('./data/PEMSBAY/PEMSBAY.npz')
        data = np.load(data_path)['data'][:, :, 0]  # traffic speed
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)

    print('Load %s Dataset shaped: ' % dataset, data.shape,
          data.max(), data.min(), data.mean(), np.median(data))
    return data
