from torch import nn
import torch


class DeepClassifier(nn.Module):
    """Deep classifier (HEAD) of the model"""

    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(64, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x


class Classifier(nn.Module):
    """Simple classifier (HEAD) of the model"""

    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred
