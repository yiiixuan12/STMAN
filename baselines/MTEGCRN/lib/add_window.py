import numpy as np

def _rolling_window_view(data, start, size, count):
    if count <= 0:
        return np.empty((0, size) + data.shape[1:], dtype=data.dtype)
    base = data[start:start + count + size - 1]
    shape = (count, size) + data.shape[1:]
    strides = (data.strides[0], data.strides[0]) + data.strides[1:]
    return np.lib.stride_tricks.as_strided(base, shape=shape, strides=strides)

def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    data = np.asarray(data)
    length = len(data)
    end_index = length - horizon - window + 1
    if single:
        X = _rolling_window_view(data, 0, window, end_index)
        Y = data[window + horizon - 1:window + horizon - 1 + max(end_index, 0), np.newaxis, ...]
    else:
        X = _rolling_window_view(data, 0, window, end_index)
        Y = _rolling_window_view(data, window, horizon, end_index)
    return X, Y

if __name__ == '__main__':
    from data.load_raw_data import Load_Sydney_Demand_Data
    path = '../data/1h_data_new3.csv'
    data = Load_Sydney_Demand_Data(path)
    print(data.shape)
    X, Y = Add_Window_Horizon(data, horizon=2)
    print(X.shape, Y.shape)


