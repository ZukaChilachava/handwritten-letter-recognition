import math
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

@dataclass
class Trainer:
    model: nn.Module
    dataset: Dataset
    learning_rate: float
    num_epochs: int = 50
    batch_size: int = 32_768
    loss_fun: nn.modules.loss = nn.CrossEntropyLoss()

    def train(self):
        steps_in_batch = math.ceil(len(self.dataset)/self.batch_size)
        dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            net_loss = 0
            for i, (labels, inputs) in enumerate(dataloader):
                predictions = self.model(inputs)

                loss = self.loss_fun(predictions, labels.long())
                net_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loading_bar(i + 1, steps_in_batch, net_loss)

        # trained_model = {
        #     "model_state": self.model.state_dict()
        # }

        # torch.save(trained_model, "model.pth")

        # save the model


def loading_bar(i, out_of, net_loss):
    progress = "#"*i + "-"*(out_of - i)
    print(f"[{progress}]", end="\r")
    if i == out_of:
        print(f"[{progress}] loss: {net_loss}")
