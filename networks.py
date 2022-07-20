import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(
        self,
        dim,
        n_hidden_layers,
        n_output=1,
        hidden_size=128,
        dropout_rate=None,
        batch_norm=False,
    ):
        super().__init__()
        layers = nn.ModuleList()

        layers.append(nn.Linear(dim, hidden_size))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.ReLU())

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, n_output))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class NetworkDropout(nn.Module):
    def __init__(self, dim, hidden_size=100, dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        return self.fc2(self.activate(self.dropout(self.fc1(x))))


class VariableNet(nn.Module):
    def __init__(self, dim, layer_widths):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(dim, layer_widths[0]))
        self.layers.append(nn.ReLU())

        for i in range(len(layer_widths) - 1):
            self.layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(layer_widths[-1], 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
