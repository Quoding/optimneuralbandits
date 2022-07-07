import json
import logging
import os
import sys
from math import ceil, floor

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import conv1d
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../..")
sys.path.append("..")
import random

from utils import *

device = torch.device("cuda")


class Network(nn.Module):
    def __init__(self, dim, n_hidden_layers, hidden_size=100, dropout_rate=None):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(dim, hidden_size))
        if dropout_rate is not None:
            self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.ReLU())

        for _ in range(n_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            if dropout_rate is not None:
                self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_size, 1))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)


class VariableNet(nn.Module):
    def __init__(self, dim, layer_widths, dropout_rate=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(dim, layer_widths[0]))
        if dropout_rate:
            self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.ReLU())

        for i in range(len(layer_widths) - 1):
            self.layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            if dropout_rate:
                self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(layer_widths[-1], 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CombiDataset(Dataset):
    def __init__(self, combis, risks):
        self.combis = combis
        self.risks = risks

    def get_weights(self, kern_size=5, kern_sigma=2, reweight="sqrt_inv"):
        bin_size = 0.1
        # Implements label distribution smoothing (Delving into Deep Imbalanced Regression, https://arxiv.org/abs/2102.09554)
        # Discretize the risks (labels used later)
        flat_risks = self.risks.flatten()
        # print(flat_risks)
        discrete_risks = torch.floor(flat_risks * 10) / 10
        discrete_risks = np.around(discrete_risks.cpu().numpy(), 1)
        # print(discrete_risks)
        # Get the gaussian filter
        kernel = gaussian_fn(kern_size, kern_sigma)[None, None]

        # Determine the bin edges
        min_bin = floor(min(flat_risks) * 10) / 10
        max_bin = ceil(max(flat_risks) * 10) / 10

        # Handles case where maximum is exactly on the edge of the last bin
        if max(flat_risks).item() == max_bin:
            max_bin += bin_size
            print(f"new max bin = {max_bin}")

        n_bins = round(
            (max_bin - min_bin) / 0.1
        )  # Deal with poor precision in Python...
        list_bin_edges = np.around(
            [min_bin + (bin_size * i) for i in range(n_bins + 1)], 1
        ).astype("float32")
        bin_edges = torch.from_numpy(list_bin_edges)

        # Build discretized distribution with histogram
        hist = torch.histogram(flat_risks.cpu(), bin_edges)
        weights = hist.hist

        # wplot = weights.cpu().numpy()
        # plt.bar(list_bin_edges[:-1], wplot, width=0.1, edgecolor="black")
        # plt.title("Distribution des risques relatifs (discret)")
        # plt.xlabel("Intervalles (0.1)")
        # plt.ylabel("Compte")
        # plt.show()

        if reweight == "sqrt_inv":
            weights = torch.sqrt(weights)

        # Apply label distribution smoothing
        weights = conv1d(weights[None, None].cuda(), kernel, padding=(kern_size // 2))
        weights = 1 / weights

        # Get weights for dataset
        weight_bins = {list_bin_edges[i]: weights[0][0][i] for i in range(n_bins)}
        weights_per_obs = [weight_bins[risk] for risk in discrete_risks]

        # k = list(weight_bins.keys())
        # v = list(weight_bins.values())
        # v = [val.item() for val in v]

        # plt.bar(k, v, width=0.1, edgecolor="black")
        # plt.title("Poids d'echantillonnage des observations")
        # plt.xlabel("Intervalles (0.1)")
        # plt.ylabel("Poids")
        # plt.show()
        return weights_per_obs

    def __len__(self):
        return len(self.risks)

    def __getitem__(self, idx):
        return self.combis[idx], self.risks[idx]


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.min_loss = float("inf")
        self.count = 0
        self.train_activ = None
        self.val_activ = None

    def __call__(self, cur_loss, train_activ, val_activ):
        # If no improvement
        if cur_loss >= self.min_loss:
            self.count += 1
        else:  # Improvement, store activs
            self.count = 0
            self.store(train_activ, val_activ)
            self.min_loss = cur_loss

    def store(self, train_activ, val_activ):
        self.train_activ = train_activ.detach().clone()
        self.val_activ = val_activ.detach().clone()

    @property
    def early_stop(self):
        if self.count >= self.patience:
            return True


def gaussian_fn(size, std):
    n = torch.arange(0, size) - (size - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-(n**2) / sig2)
    return w


# %%
def load_dataset(dataset_path):
    dataset = pd.read_csv("../datasets/combinations/" + dataset_path + ".csv")

    with open("../datasets/patterns/" + dataset_path + ".json", "r") as f:
        patterns = json.load(f)

    features = dataset.iloc[:, :-3]
    risks = dataset.iloc[:, -3]

    n_obs, n_dim = features.shape

    return features, risks, patterns, n_obs, n_dim


def setup_data(dataset, batch_size, n_obs, reweight=None):
    combis, risks, _, _, n_dim = load_dataset(dataset)

    combis, risks = (
        torch.tensor(combis.values).float(),
        torch.tensor(risks.values).unsqueeze(1).float(),
    )

    X_train, y_train = combis[:n_obs], risks[:n_obs]
    X_val, y_val = combis[n_obs:], risks[n_obs:]

    training_data = CombiDataset(X_train, y_train)
    if reweight is not None:
        print("Using label distribution smoothing")
        w = training_data.get_weights(reweight=reweight)
        sampler = WeightedRandomSampler(w, num_samples=n_obs)
        trainloader = DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=False,
            generator=torch.Generator(device="cuda"),
            sampler=sampler,
        )
    else:
        trainloader = DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator(device="cuda"),
        )

    return trainloader, X_train, y_train, X_val, y_val, n_dim


def plot_losses(n_epochs, train_losses, val_losses):
    fig = plt.figure()
    x = list(range(n_epochs))
    train_nanmean = np.nanmean(train_losses, axis=0)
    train_nanstd = np.nanstd(train_losses, axis=0)
    val_nanmean = np.nanmean(val_losses, axis=0)
    val_nanstd = np.nanstd(val_losses, axis=0)

    # plot training loss
    plt.plot(x, train_nanmean, label="Entrainement", color="tab:blue")
    plt.fill_between(
        x,
        train_nanmean - train_nanstd,
        train_nanmean + train_nanstd,
        alpha=0.3,
        color="tab:blue",
    )

    # plot val loss
    plt.plot(x, val_nanmean, label="Validation", color="tab:orange")
    plt.fill_between(
        x,
        val_nanmean - val_nanstd,
        val_nanmean + val_nanstd,
        alpha=0.3,
        color="tab:orange",
    )

    plt.title("Pertes selon le nombre d'époques")
    plt.xlabel("Époque")
    plt.ylabel("Perte")
    plt.ylim(0, 1.2)
    plt.xlim(0, n_epochs)
    plt.yticks(np.arange(0, 1 + 0.05, 0.05))
    plt.xticks(np.arange(0, n_epochs, 1))
    plt.legend()

    return fig


def plot_pred_vs_gt(true, pred, title):
    fig = plt.figure()
    plt.title(title)
    plt.xlabel("Vérité")
    plt.ylabel("Prédiction")
    plt.ylim(0, 3.5)
    plt.xlim(0, 3.5)
    plt.scatter(true, pred, alpha=0.1)
    plt.plot([0, 3], [0, 3], color="black", linestyle="dashed", label="Perfection")
    plt.legend()

    return fig


def save_metrics(array, path):
    dir_ = "/".join(path.split("/")[:-1])
    os.makedirs(dir_, exist_ok=True)
    np.save(path, array)


def get_loss(loss_name):
    if loss_name == "mse" or loss_name == "rmse":
        return nn.MSELoss()
    elif loss_name == "l1":
        return nn.L1Loss()
