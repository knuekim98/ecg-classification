import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


class ECGDataset(Dataset):
    def __init__(self, x, y):
        super(ECGDataset, self).__init__()

        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.y)


def read_dataset(train_batch_size, test_batch_size):
    train = pd.read_csv("./dataset/mitbih_train.csv", header=None).values
    test = pd.read_csv("./dataset/mitbih_test.csv", header=None).values

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]
    

    train_dataset = ECGDataset(X_train, y_train)
    test_dataset = ECGDataset(X_test, y_test)

    VAL_DATA_RATIO = 0.2
    val_data_len = int(len(train_dataset)*VAL_DATA_RATIO)
    train_subset, val_subset = random_split(train_dataset, [len(train_dataset)-val_data_len, val_data_len], generator=torch.Generator().manual_seed(42))
    

    train_loader = DataLoader(train_subset, train_batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, test_batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
    

if __name__ == "__main__":
    train_loader, _, _ = read_dataset(16, 100)
    for i, (X, y) in enumerate(train_loader):
        print(X.shape)
        sample = X[0]
        plt.plot(np.linspace(0, 10, len(sample)), sample)
        plt.show()

        break