import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 100)

    def forward(self, x):
        return F.relu(self.linear(x))

neural = NN()

print(neural.parameters())