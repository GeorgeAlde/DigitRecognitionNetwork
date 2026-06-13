import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_train, y_train, X_test, y_test = data.get_data()
X_train = torch.tensor(X_train, device = device)
y_train = torch.tensor(y_train, device = device)
X_test = torch.tensor(X_test, device = device)
y_test = torch.tensor(y_test, device = device)


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

model = NeuralNetwork(784, 10).to(device)
learning_rate = 0.001
optimiser = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

epochs = 1000
for epoch in range(epochs):
    model.train()
    y_hat = model(X_train)
    loss = loss_fn(y_hat, y_train)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

model.eval()
with torch.no_grad():
    m = nn.Softmax(dim=1)
    train_outputs = model(X_train)
    train_predictions = (m(train_outputs)).float()

    train_accuracy = ((torch.argmax(train_predictions, dim=1) == y_train).float().mean())

    test_outputs = model(X_test)

    test_predictions = (m(test_outputs)).float()

    test_accuracy = ((torch.argmax(test_predictions, dim=1) == y_test).float().mean())

print(f"Train Accuracy: {train_accuracy.item()*100:.2f}%")
print(f"Test Accuracy : {test_accuracy.item()*100:.2f}%")

for i in range(10):
    plt.imshow(X_test[i].cpu().reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.show()
    print(f"Predicted value: {torch.argmax(test_predictions, dim=1)[i]}")
    plt.close()