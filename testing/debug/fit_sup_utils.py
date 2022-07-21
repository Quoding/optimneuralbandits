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
import random

sys.path.append("../..")
from utils import *


device = torch.device("cuda")


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

            early_stopping(val_loss, train_activ, val_activ, None)

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
            # print(weight_bins)
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
        self.test_activ = None

    def __call__(self, cur_loss, train_activ, val_activ, test_activ):
        # If no improvement
        if cur_loss >= self.min_loss:
            self.count += 1
        else:  # Improvement, store activs
            self.count = 0
            self.store(train_activ, val_activ, test_activ)
            self.min_loss = cur_loss

    def store(self, train_activ, val_activ, test_activ):
        self.train_activ = train_activ.detach().clone().cpu().numpy()
        self.val_activ = val_activ.detach().clone().cpu().numpy()
        if test_activ is not None:
            self.test_activ = test_activ.detach().clone().cpu().numpy()

    @property
    def early_stop(self):
        if self.count >= self.patience:
            return True


def load_dataset(dataset_path):
    dataset = pd.read_csv("../datasets/combinations/" + dataset_path + ".csv")

    with open("../datasets/patterns/" + dataset_path + ".json", "r") as f:
        patterns = json.load(f)

    features = dataset.iloc[:, :-3]
    risks = dataset.iloc[:, -3]

    n_obs, n_dim = features.shape

    return features, risks, patterns, n_obs, n_dim


def setup_data(
    dataset, batch_size, n_obs, reweight=None, classif_thresh=None, val=None
):
    combis, risks, _, _, n_dim = load_dataset(dataset)

    combis, risks = (
        torch.tensor(combis.values).float(),
        torch.tensor(risks.values).unsqueeze(1).float(),
    )

    # Shuffle the data
    perm_idx = torch.randperm(len(risks))
    combis = combis[perm_idx]
    risks = risks[perm_idx]

    if val == "extrema":
        # Use extrema for validation set (min and max)
        min_idx = torch.argmin(risks)
        max_idx = torch.argmax(risks)
        ids = torch.tensor([min_idx, max_idx])
        X_val = combis[ids]
        y_val = risks[ids]
        min_of_indexes = min(min_idx, max_idx)
        max_of_indexes = max(min_idx, max_idx)

        # Remove used indexes from the observations
        combis = torch.cat((combis[:max_of_indexes], combis[max_of_indexes + 1 :]))
        combis = torch.cat((combis[:min_of_indexes], combis[min_of_indexes + 1 :]))
        risks = torch.cat((risks[:max_of_indexes], risks[max_of_indexes + 1 :]))
        risks = torch.cat((risks[:min_of_indexes], risks[min_of_indexes + 1 :]))

        # Create a test set so we can see how good the model actually is different stages
        X_test, y_test = combis[n_obs:], risks[n_obs:]
    elif val == "bins":
        # Use bins for validation set, so we have a wide spread
        bin_size = 0.1
        min_range = 0
        max_range = 4
        bins_left_edge = [
            round(min_range + (i * bin_size), 1)
            for i in range(int(max_range / bin_size) + 1)
        ]
        X_val = []
        y_val = []
        indices = []
        # Put one observation per bin
        for i in range(len(bins_left_edge) - 1):
            lower_bound = bins_left_edge[i]
            upper_bound = bins_left_edge[i + 1]
            bigger_than = risks >= lower_bound
            smaller_than = risks < upper_bound
            in_bounds = torch.cat((bigger_than, smaller_than), dim=1).all(dim=1)
            idx = torch.where(in_bounds)[0]
            if len(idx) > 0:
                indices.append(idx[0].item())

        mask = torch.ones(len(risks)).bool()
        mask[indices] = False

        X_val = combis[indices]
        y_val = risks[indices]
        combis = combis[mask]
        risks = risks[mask]

        # Create a test set so we can see how good the model actually is different stages
        X_test, y_test = combis[n_obs:], risks[n_obs:]
    else:
        X_val, y_val = combis[n_obs:], risks[n_obs:]
        X_test, y_test = None, None

    X_train, y_train = combis[:n_obs], risks[:n_obs]

    if classif_thresh is not None:
        y_train = (y_train > classif_thresh).float()
        y_val = (y_val > classif_thresh).float()

    training_data = CombiDataset(X_train, y_train, classif_thresh)
    if reweight is not None:
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

    return trainloader, training_data, X_val, y_val, n_dim, X_test, y_test


def plot_metric(n_epochs, train_losses, val_losses, test_losses=None):
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

    # plot test loss
    if test_losses is not None:
        test_nanmean = np.nanmean(test_losses, axis=0)
        test_nanstd = np.nanstd(test_losses, axis=0)
        plt.plot(x, test_nanmean, label="Test", color="tab:green")
        plt.fill_between(
            x,
            test_nanmean - test_nanstd,
            test_nanmean + test_nanstd,
            alpha=0.3,
            color="tab:green",
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


def plot_pred_vs_gt(
    train_true,
    train_pred,
    val_true,
    val_pred,
    test_true,
    test_pred,
    title,
    pred_idx,
    invert=False,
):
    fig = plt.figure()
    plt.title(title)
    plt.xlabel("Vérité")
    plt.ylabel("Prédiction")
    ylim = 3.5
    xlim = 3.5
    plt.ylim(0, ylim)
    plt.xlim(0, xlim)
    val_alpha = 0.1
    z_val = 1
    z_test = 1
    z_train = 1
    if invert:
        z_test = 2

    # If we have a test set (basically, if we have a small validation set), plot it
    if test_true is not None and test_pred is not None:
        plt.scatter(
            test_true.cpu().numpy(),
            test_pred[:, pred_idx],
            alpha=0.1,
            color="tab:green",
            label="Test",
            zorder=z_test,
        )
        # Set alpha of small validation set to be more visible
        val_alpha = 1
        z_val = 10

    plt.scatter(
        val_true.cpu().numpy(),
        val_pred[:, pred_idx],
        alpha=val_alpha,
        color="tab:orange",
        label="Validation",
        zorder=z_val,
    )
    plt.scatter(
        train_true.cpu().numpy(),
        train_pred[:, pred_idx],
        alpha=0.1,
        color="tab:blue",
        label="Entrainement",
        zorder=z_train,
    )

    plt.plot(
        [0, xlim], [0, ylim], color="black", linestyle="dashed", label="Perfection"
    )

    plt.plot(
        [1.1, 1.1], [0, 1.1], color="black", linestyle="dotted", label="Seuil de risque"
    )
    plt.plot([0, 1.1], [1.1, 1.1], color="black", linestyle="dotted")
    plt.legend()

    return fig


def save_metrics(array, path):
    dir_ = "/".join(path.split("/")[:-1])
    os.makedirs(dir_, exist_ok=True)
    np.save(path, array)


def get_loss(loss_name, quantiles=[0.5, 0.7]):
    if loss_name == "mse" or loss_name == "rmse":
        return nn.MSELoss()
    elif loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "quantile":
        return QuantileLoss(quantiles)


def get_layers(n_dim, embed_dim):
    layers = []
    d = embed_dim
    while d < n_dim:
        if d != n_dim and d != embed_dim:
            layers.append(d)
        d = 1 << (d).bit_length()
    layers.reverse()
    return layers


def get_losses_and_activ(
    net,
    criterion,
    loss_info,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    transform=nn.Identity(),
):
    val_activ = net(transform(X_val))
    train_activ = net(transform(X_train))

    val_loss = criterion(val_activ, y_val)
    train_loss = criterion(train_activ, y_train)

    if loss_info[0] == "rmse":
        train_loss = torch.sqrt(train_loss)
        val_loss = torch.sqrt(val_loss)

    val_loss = val_loss.item()
    train_loss = train_loss.item()

    if X_test is not None and y_test is not None:
        test_activ = net(transform(X_test))
        test_loss = criterion(test_activ, y_test)
        if loss_info[0] == "rmse":
            test_loss = torch.sqrt(test_loss)
        test_loss = test_loss.item()
    else:
        test_activ, test_loss = None, None

    return train_activ, train_loss, val_activ, val_loss, test_activ, test_loss


def update_minimums(
    train_loss,
    min_train_loss,
    train_activ,
    val_loss,
    min_val_loss,
    val_activ,
    test_loss,
    test_activ,
    pred_idx,
    val_activ_min_loss,
    train_activ_min_loss,
    test_activ_min_loss,
    val_activ_mintrain_loss,
    train_activ_mintrain_loss,
    test_activ_mintrain_loss,
):
    # Record all activations on new minimum val loss
    if val_loss < min_val_loss:
        val_activ_min_loss = val_activ.detach().clone().cpu().numpy()
        train_activ_min_loss = train_activ.detach().clone().cpu().numpy()
        min_val_loss = val_loss
        if test_loss is not None and test_activ is not None:
            test_activ_min_loss = test_activ.detach().clone().cpu().numpy()

    # Record all activations on new minimum train loss
    if train_loss < min_train_loss:
        val_activ_mintrain_loss = val_activ.detach().clone().cpu().numpy()
        train_activ_mintrain_loss = train_activ.detach().clone().cpu().numpy()
        min_train_loss = train_loss
        if test_loss is not None and test_activ is not None:
            test_activ_mintrain_loss = test_activ.detach().clone().cpu().numpy()
    return (
        train_activ_min_loss,
        val_activ_min_loss,
        test_activ_min_loss,
        min_val_loss,
        train_activ_mintrain_loss,
        val_activ_mintrain_loss,
        test_activ_mintrain_loss,
        min_train_loss,
    )
