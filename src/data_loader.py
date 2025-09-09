import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

class HAR_Dataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

def _load_file(filepath):
    df = pd.read_csv(filepath, header=None, sep=r'\s+')
    return df.values

def _load_group(filenames, prefix=''):
    loaded = [ _load_file(os.path.join(prefix, name)) for name in filenames ]
    return np.dstack(loaded)

def get_dataloaders(base_path, batch_size=64):
    train_signals_path = os.path.join(base_path, 'train/Inertial Signals/')
    x_train = _load_group(['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt',
                           'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
                           'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt'],
                          train_signals_path)
    y_train = _load_file(os.path.join(base_path, 'train/y_train.txt'))

    test_signals_path = os.path.join(base_path, 'test/Inertial Signals/')
    x_test = _load_group(['body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt',
                         'body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt',
                         'total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt'],
                         test_signals_path)
    y_test = _load_file(os.path.join(base_path, 'test/y_test.txt'))

    y_train = y_train.flatten() - 1
    y_test = y_test.flatten() - 1
    
    x_train = x_train.transpose(0, 2, 1)
    x_test = x_test.transpose(0, 2, 1)

    train_dataset = HAR_Dataset(x_train, y_train)
    test_dataset = HAR_Dataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader