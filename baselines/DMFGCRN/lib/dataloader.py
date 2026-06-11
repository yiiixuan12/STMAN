import torch
import numpy as np
import torch.utils.data
from datetime import datetime, timedelta
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

DATASET_START_DATES = {
    'PEMS03': '2018-09-01',
    'PEMS04': '2018-01-01',
    'PEMS07': '2017-05-01',
    'PEMS08': '2016-07-01',
    'PEMSD3': '2018-09-01',
    'PEMSD4': '2018-01-01',
    'PEMSD7': '2017-05-01',
    'PEMSD8': '2016-07-01',
    'METRLA':'2012-03-01',
    'PEMSBAY': '2017-01-01',
    'Vehicular': '2022-08-28',
    'Pedestrian': '2022-08-28',
    'Intersection': '2022-08-28',
}


def get_time_features(dataset_name, time_stamps, num_nodes, steps_per_day=288, interval_minutes=5):
    if dataset_name == 'Intersection':
        steps_per_day = 96
        interval_minutes = 15

    if dataset_name in DATASET_START_DATES:
        start_date = datetime.strptime(DATASET_START_DATES[dataset_name], "%Y-%m-%d")
    else:
        print(f"Warning: Dataset {dataset_name} not in date mapping, using default date")
        start_date = datetime(2020, 1, 1)  

    time_in_day = np.zeros((time_stamps, num_nodes, 1), dtype=np.float32)
    day_in_week = np.zeros((time_stamps, num_nodes, 1), dtype=np.float32)

    current_date = start_date

    for t in range(time_stamps):

        time_in_day[t, :, 0] = (t % steps_per_day) / steps_per_day

        day_in_week[t, :, 0] = current_date.weekday() / 7.0  

        current_date = current_date + timedelta(minutes=interval_minutes)

    return time_in_day, day_in_week


def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError(f"Unknown normalizer: {normalizer}")
    return scaler


def split_data_by_days(data, val_days, test_days, interval=30):
    """按天数划分数据"""
    T = int((24 * 60) / interval)
    test_data = data[-T * int(test_days):]
    val_data = data[-T * int(test_days + val_days): -T * int(test_days)]
    train_data = data[:-T * int(test_days + val_days)]
    return train_data, val_data, test_data


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    X = torch.from_numpy(np.asarray(X, dtype=np.float32))
    Y = torch.from_numpy(np.asarray(Y, dtype=np.float32))
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(args, normalizer='std', tod=False, dow=False, weather=False, single=True):
    data = np.asarray(load_st_dataset(args.dataset), dtype=np.float32)  # (L, N, F)
    L, N, F = data.shape

    interval_minutes = int(24 * 60 / args.steps_per_day)

    time_in_day, day_in_week = get_time_features(
        dataset_name=args.dataset,
        time_stamps=L,
        num_nodes=N,
        steps_per_day=args.steps_per_day,
        interval_minutes=interval_minutes
    )

    x, y = Add_Window_Horizon(data, args.lag, args.horizon, single)
    x_day, y_day = Add_Window_Horizon(time_in_day, args.lag, args.horizon, single)
    x_week, y_week = Add_Window_Horizon(day_in_week, args.lag, args.horizon, single)
    x = np.concatenate([x, x_day, x_week], axis=-1).astype(np.float32, copy=False)
    y = np.asarray(y, dtype=np.float32)
    # Only x needs calendar features; training/evaluation uses the first
    # output_dim channel from y as the label. Keeping y_day/y_week here
    # triples long-horizon label memory without changing the loss.

    if args.test_ratio > 1:
        x_train, x_val, x_test = split_data_by_days(x, args.val_ratio, args.test_ratio, interval_minutes)
        y_train, y_val, y_test = split_data_by_days(y, args.val_ratio, args.test_ratio, interval_minutes)
    else:
        x_train, x_val, x_test = split_data_by_ratio(x, args.val_ratio, args.test_ratio)
        y_train, y_val, y_test = split_data_by_ratio(y, args.val_ratio, args.test_ratio)

    scaler = normalize_dataset(x_train[..., :args.input_dim], normalizer, args.column_wise)
    x_train[..., :args.input_dim] = scaler.transform(x_train[..., :args.input_dim])
    x_val[..., :args.input_dim] = scaler.transform(x_val[..., :args.input_dim])
    x_test[..., :args.input_dim] = scaler.transform(x_test[..., :args.input_dim])

    print('Train: ', x_train.shape, y_train.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    train_dataloader = data_loader(x_train, y_train, args.batch_size, shuffle=True, drop_last=True)

    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)

    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler


def get_advanced_time_features(dataset_name, time_stamps, num_nodes, steps_per_day=288, interval_minutes=5):
    if dataset_name in DATASET_START_DATES:
        start_date = datetime.strptime(DATASET_START_DATES[dataset_name], "%Y-%m-%d")
    else:
        start_date = datetime(2020, 1, 1)

    time_features = np.zeros((time_stamps, num_nodes, 5))  # 5个时间特征
    current_date = start_date

    for t in range(time_stamps):
        # 1. Time of Day (归一化)
        time_features[t, :, 0] = (t % steps_per_day) / steps_per_day

        # 2. Day of Week (归一化)
        time_features[t, :, 1] = current_date.weekday() / 7.0

        # 3. Hour of Day (归一化到[0,1])
        hour = (t % steps_per_day) * 24 / steps_per_day
        time_features[t, :, 2] = hour / 24.0

        # 4. Is Weekend (0 or 1)
        time_features[t, :, 3] = 1.0 if current_date.weekday() >= 5 else 0.0

        # 5. Month (归一化到[0,1])
        time_features[t, :, 4] = (current_date.month - 1) / 11.0

        current_date = current_date + timedelta(minutes=interval_minutes)

    return time_features

def get_adjacency_matrix2(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


if __name__ == '__main__':
    import argparse
    #MetrLA 207; BikeNYC 128; SIGIR_solar 137; SIGIR_electric 321
    DATASET = 'SIGIR_electric'
    if DATASET == 'MetrLA':
        NODE_NUM = 207
    elif DATASET == 'BikeNYC':
        NODE_NUM = 128
    elif DATASET == 'SIGIR_solar':
        NODE_NUM = 137
    elif DATASET == 'SIGIR_electric':
        NODE_NUM = 321
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True)
