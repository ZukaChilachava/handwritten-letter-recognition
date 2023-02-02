import sys
import math
import torch
import numpy as np
import os.path as pt
from trainer import Trainer
from dataset import LetterDataset
from letter_models import LetterFNN, LetterCNN


def test(dataset: LetterDataset, model):
    labels, inputs = dataset.y, dataset.x

    with torch.no_grad():
        prediction_magnitudes = model(inputs)

        predictions = torch.argmax(prediction_magnitudes, 1)
        print(predictions, labels)
        correctly_predicted = (predictions == labels).sum().item()

    print(f"Accuracy: {100*correctly_predicted/len(dataset)}")


def load_saved():
    saved_model = torch.load("model.pth")["model_state"]
    ffn = LetterFNN(train_dataset[0][1].shape[0], num_letters)
    ffn.load_state_dict(saved_model)

    return ffn

if __name__ == '__main__':
    path_to_root = sys.path[1]
    path_to_dataset = pt.join(path_to_root, "Data", "A_Z Handwritten Data.csv")

    raw_data = np.loadtxt(fname=path_to_dataset, dtype=np.float32, delimiter=",")
    np.random.shuffle(raw_data)
    print("Data loaded")

    num_letters = 26
    learning_rate = 0.001
    total_samples = raw_data.shape[0]
    training_samples = math.floor(total_samples*0.8)

    train_dataset = LetterDataset(raw_data, range(0, training_samples))
    test_dataset = LetterDataset(raw_data, range(training_samples, total_samples))

    # FNN
    ffn_model = LetterFNN(train_dataset[0][1].shape[0], num_letters)
    ffn_trainer = Trainer(ffn_model, train_dataset, learning_rate)
    ffn_trainer.train()
    test(test_dataset, ffn_model)

    # CNN
    # train_dataset.x = torch.reshape(train_dataset.x, (training_samples, 1, 28, 28))
    # cnn_model = LetterCNN(num_letters)
    # cnn_trainer = Trainer(cnn_model, train_dataset, learning_rate, batch_size=65_536)
    # cnn_trainer.train()

