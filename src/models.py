import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.0, use_batch_norm=False):
        super(FlexibleNN, self).__init__()

        layers = []
        in_size = input_size

        previous_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(previous_size, size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            previous_size = size

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(previous_size, output_size)

    def forward(self, x):
        print(x.shape)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x