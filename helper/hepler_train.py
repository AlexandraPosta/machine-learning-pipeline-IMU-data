import torch
import torch.nn as nn
import torch.optim as optim

# ANN model
# Define the neural network model
class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# SVM model


# CNN model