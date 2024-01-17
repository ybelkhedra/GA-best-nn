import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.0, use_batch_norm=False):
        super(FlexibleNN, self).__init__()

        layers = []
        in_size = input_size

        for out_size in hidden_sizes:
            layers.append(nn.Linear(in_size, out_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            in_size = out_size

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_size, output_size)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x