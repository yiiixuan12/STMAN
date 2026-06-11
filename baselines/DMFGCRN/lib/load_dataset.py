import os
import numpy as np
import pandas as pd

def load_st_dataset(dataset):
    #output B, N, D
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
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD4':
        data_path = os.path.join('./data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7':
        data_path = os.path.join('./data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('./data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7(L)':
        data_path = os.path.join('./data/PEMSD7L/PEMSD7L.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7(M)':
        data_path = os.path.join('./data/PEMSD7M/PEMSD7M.npz')
        data = np.load(data_path)['data'][:, :, 0]
        # data = np.array(pd.read_csv(data_path,header=None))  #onley the first dimension, traffic flow data
    elif dataset == 'bike_pick':
        data_path = os.path.join('./data/bike_pick.npz')
        data = np.load(data_path)['data'][:, :, 0]
        # data = np.array(pd.read_csv(data_path,header=None))  #onley the first dimension, traffic flow data
    elif dataset == 'bike_drop':
        data_path = os.path.join('./data/bike_drop.npz')
        data = np.load(data_path)['data'][:, :, 0]
        # data = np.array(pd.read_csv(data_path,header=None))  #onley the first dimension, traffic flow data
    elif dataset == 'METRLA':
        data_path = os.path.join('./data/METRLA/METRLA.npz')
        data = np.load(data_path)['data'][:, :, 0]
    elif dataset == 'PEMSBAY':
        data_path = os.path.join('./data/PEMSBAY/PEMSBAY.npz')
        # Print all available keys
        # First, load and inspect the NPZ file
        # npz_file = np.load(data_path)
        # print("Available keys in the NPZ file:")
        # print(npz_file.files)  # or list(npz_file.keys())

        # # Check the shape and type of each array
        # for key in npz_file.files:
        #     print(f"{key}: shape={npz_file[key].shape}, dtype={npz_file[key].dtype}")
        data = np.load(data_path)['data'][:, :, 0]
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

#
# data_path = os.path.join('../data/PeMS07/PEMS07.npz')
# data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
