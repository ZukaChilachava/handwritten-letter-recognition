import torch
from torch.utils.data import Dataset


class LetterDataset(Dataset):
    def __init__(self, raw_data, data_range):
        xy = raw_data[data_range]
        self.y = torch.as_tensor(xy[:, 0])
        self.x = torch.as_tensor(xy[:, 1:])
        self.n_samples = xy.shape[0]

    def __getitem__(self, item):
        return self.y[item], self.x[item]

    def __len__(self):
        return self.n_samples
