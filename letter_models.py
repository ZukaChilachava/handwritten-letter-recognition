import torch.nn as nn
import torch.nn.functional as F

class LetterFNN(nn.Module):

    def __init__(self, input_size, num_classes):
        super(LetterFNN, self).__init__()
        self.relu = nn.ReLU()
        self.L1 = nn.Linear(in_features=input_size, out_features=512)
        self.L2 = nn.Linear(in_features=512, out_features=256)
        self.L3 = nn.Linear(in_features=256, out_features=128)
        self.L4 = nn.Linear(in_features=128, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        L1 = self.L1(x)
        ReLU1 = self.relu(L1)
        L2 = self.L2(ReLU1)
        ReLU2 = self.relu(L2)
        L3 = self.L3(ReLU2)
        ReLU3 = self.relu(L3)
        L4 = self.L4(ReLU3)
        ReLU4 = self.relu(L4)
        out = self.out(ReLU4)

        return out


class LetterCNN(nn.Module):

    def __init__(self, num_letters):
        super(LetterCNN, self).__init__()
        self.relu = nn.ReLU()
        # 24x24
        self.conv = nn.Conv2d(1, 3, 5)
        # 12x12
        self.pool = nn.MaxPool2d(2, 2)
        self.L1 = nn.Linear(in_features=432, out_features=256)
        self.L2 = nn.Linear(in_features=256, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=num_letters)

    def forward(self, x):
        conv = self.conv(x)
        ReLU1 = self.relu(conv)
        pool = self.pool(ReLU1)
        pool = pool.view(-1, 3*12*12)
        L1 = self.L1(pool)
        ReLU2 = self.relu(L1)
        L2 = self.L2(ReLU2)
        ReLU3 = self.relu(L2)
        out = self.out(ReLU3)

        return out
