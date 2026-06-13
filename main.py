import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_train, y_train, X_test, y_test = data.get_data()
X_train = torch.tensor(X_train, device = device)
y_train = torch.tensor(y_train, device = device)
X_test = torch.tensor(X_test, device = device)
y_test = torch.tensor(y_test, device = device)

print(X_train[0].shape)

class NeuralNetwork(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, 15),
            nn.ReLU(),
            nn.Linear(15, outputs)
        )
    def forward(self, x):
        return self.network(x)

model = NeuralNetwork(784, 10)
learning_rate = 0.01
