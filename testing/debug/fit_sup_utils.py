import json
import logging
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import conv1d
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import _Loss

sys.path.append("../..")
sys.path.append("..")
import random

from utils import *

device = torch.device("cuda")


class Network(nn.Module):
    def __init__(
        self, dim, n_hidden_layers, hidden_size=100, dropout_rate=None, batch_norm=False
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

        layers.append(nn.Linear(hidden_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AE(nn.Module):
    def __init__(self, input_dim, embed_dim, layer_widths):
        super().__init__()

        self.encoder = self.create_encoder(input_dim, embed_dim, layer_widths)
        self.decoder = self.create_decoder(input_dim, embed_dim, layer_widths)

    def create_encoder(self, input_dim, embed_dim, layer_widths):
        if len(layer_widths) == 0:
            return nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ReLU())

        # Define encoder
        enc_layers = nn.ModuleList()

        enc_layers.append(nn.Linear(input_dim, layer_widths[0]))
        enc_layers.append(nn.ReLU())

        for i in range(len(layer_widths) - 1):
            enc_layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            enc_layers.append(nn.ReLU())

        enc_layers.append(nn.Linear(layer_widths[-1], embed_dim))
        enc_layers.append(nn.ReLU())

        return nn.Sequential(*enc_layers)

    def create_decoder(self, input_dim, embed_dim, layer_widths):
        if len(layer_widths) == 0:
            return nn.Sequential(nn.Linear(embed_dim, input_dim), nn.Sigmoid())

        # Definer decoder
        dec_layers = nn.ModuleList()

        dec_layers.append(nn.Linear(embed_dim, layer_widths[-1]))
        dec_layers.append(nn.ReLU())

        for i in range(len(layer_widths) - 1, 0, -1):
            dec_layers.append(nn.Linear(layer_widths[i], layer_widths[i - 1]))
            dec_layers.append(nn.ReLU())

        dec_layers.append(nn.Linear(layer_widths[0], input_dim))
        dec_layers.append(nn.Sigmoid())

        return nn.Sequential(*dec_layers)

    def fit(
        self,
        n_epochs,
        training_data,
        X_val,
        writer,
        lr=0.01,
        batch_size=32,
        patience=5,
    ):
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        trainloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
        X_train = training_data.combis
        early_stopping = EarlyStoppingActiv(patience=patience)
        for e in range(n_epochs):
            for X, _ in trainloader:
                optim.zero_grad()
                rec = self.forward(X)
                l = criterion(rec, X)
                l.backward()
                optim.step()
            with torch.no_grad():
                val_activ = self.forward(X_val)
                val_loss = criterion(val_activ, X_val)

                train_activ = self.forward(X_train)
                train_loss = criterion(train_activ, X_train)

                val_loss = val_loss.item()
                train_loss = train_loss.item()

                writer.add_scalar("Loss_ae/train", train_loss, e)
                writer.add_scalar("Loss_ae/val", val_loss, e)

            early_stopping(val_loss, train_activ, val_activ)

            if early_stopping.early_stop:
                return

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VariableNet(nn.Module):
    def __init__(self, dim, layer_widths, dropout_rate=False):
        super().__init__()
        layers = nn.ModuleList()
        layers.append(nn.Linear(dim, layer_widths[0]))
        if dropout_rate:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.ReLU())

        for i in range(len(layer_widths) - 1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(layer_widths[-1], 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CombiDataset(Dataset):
    def __init__(self, combis, labels, classif_thresh=None):
        self.combis = combis
        self.labels = labels

    def get_weights(
        self, kern_size=5, kern_sigma=2, reweight="sqrt_inv", classif_thresh=None
    ):
        if classif_thresh is None:
            bin_size = 0.1
            factor = 10
            # Implements label distribution smoothing (Delving into Deep Imbalanced Regression, https://arxiv.org/abs/2102.09554)
            # Discretize the risks (labels used later)
            flat_labels = self.labels.flatten()
            discrete_risks = discretize_targets(flat_labels, factor)

            hist, n_bins, list_bin_edges = build_histogram(
                flat_labels, factor, bin_size
            )
            weights = hist.hist

            # wplot = weights.cpu().numpy()
            # plt.bar(list_bin_edges[:-1], wplot, width=0.1, edgecolor="black")
            # plt.title("Distribution des risques relatifs (discret)")
            # plt.xlabel("Intervalles (0.1)")
            # plt.ylabel("Compte")
            # plt.show()

            if reweight == "sqrt_inv":
                weights = torch.sqrt(weights)

            # Apply label distribution smoothing with gaussian filter
            # Get the gaussian filter
            kernel = gaussian_fn(kern_size, kern_sigma)[None, None]
            weights = conv1d(
                weights[None, None].cuda(), kernel, padding=(kern_size // 2)
            )
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
        else:
            n_obs = self.labels.shape[0]
            n_pos = self.labels.sum()
            n_neg = n_obs - n_pos

            # Weights inversely proportional to the number of observations for that label
            weight_pos = 1 / (n_pos / n_obs)
            weight_neg = 1 / (n_neg / n_obs)

            weights = torch.where(self.labels == 1.0, weight_pos, weight_neg)
            return weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.combis[idx], self.labels[idx]


class EarlyStoppingActiv:
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


def setup_data(dataset, batch_size, n_obs, reweight=None, classif_thresh=None):
    combis, risks, _, _, n_dim = load_dataset(dataset)

    combis, risks = (
        torch.tensor(combis.values).float(),
        torch.tensor(risks.values).unsqueeze(1).float(),
    )

    X_train, y_train = combis[:n_obs], risks[:n_obs]
    X_val, y_val = combis[n_obs:], risks[n_obs:]

    if classif_thresh is not None:
        y_train = (y_train > classif_thresh).float()
        y_val = (y_val > classif_thresh).float()

    training_data = CombiDataset(X_train, y_train, classif_thresh)
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

    return trainloader, training_data, X_val, y_val, n_dim


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


def plot_pred_vs_gt(train_true, train_pred, val_true, val_pred, title):
    fig = plt.figure()
    plt.title(title)
    plt.xlabel("Vérité")
    plt.ylabel("Prédiction")
    ylim = 3.5
    xlim = 3.5
    plt.ylim(0, ylim)
    plt.xlim(0, xlim)
    plt.scatter(val_true, val_pred, alpha=0.1, color="tab:orange", label="Validation")
    plt.scatter(
        train_true, train_pred, alpha=0.1, color="tab:blue", label="Entrainement"
    )
    plt.plot(
        [0, xlim], [0, ylim], color="black", linestyle="dashed", label="Perfection"
    )
    plt.legend()

    return fig


def save_metrics(array, path):
    dir_ = "/".join(path.split("/")[:-1])
    os.makedirs(dir_, exist_ok=True)
    np.save(path, array)


class QuantileLoss(_Loss):
    def __init__(self, quantile):
        super().__init__()
        self.quantile = quantile

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        errors = target - preds
        loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        loss = torch.mean(loss)
        return loss


def get_loss(loss_name, quantile=0.9):
    if loss_name == "mse" or loss_name == "rmse":
        return nn.MSELoss()
    elif loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "quantile":
        return QuantileLoss(quantile)


def get_layers(n_dim, embed_dim):
    layers = []
    d = embed_dim
    while d < n_dim:
        if d != n_dim and d != embed_dim:
            layers.append(d)
        d = 1 << (d).bit_length()
    layers.reverse()
    return layers
