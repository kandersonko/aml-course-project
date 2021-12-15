import numpy as np
import scipy.io as sio
import torch

from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader



def load_data(data_dir, batch_size):
    train_data = sio.loadmat(data_dir+'trainingset.mat')
    X_train = train_data['trainset'].astype(np.float32)
    y_train = train_data['traintarget'].astype(np.float32)
    X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42)
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_val = torch.tensor(X_val)
    y_val = torch.tensor(y_val)
    y_train = y_train.argmax(axis=-1)
    y_val = y_val.argmax(axis=-1)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16)
    return train_loader, val_loader

