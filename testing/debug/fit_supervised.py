# %% [markdown]
# # Imports

# %%
import json
import logging
import sys
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../..")
sys.path.append("..")
from utils import *
from tqdm import tqdm
import random
from itertools import product


matplotlib.use("Agg")
# %% [markdown]
# # Globals

# %%
plt.rcParams["figure.figsize"] = (7, 5)  # default = (6.4, 4.8)
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 140  # default = 100
plt.rcParams["font.family"] = "serif"
# plt.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
plt.style.use("ggplot")
title_font_size = "10"

THRESH = 1.1
device = torch.device("cuda")

# %% [markdown]
# # Utility classes and functions

# %%
class Network(nn.Module):
    def __init__(self, dim, n_hidden_layers, hidden_size=100):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(dim, hidden_size))
        self.layers.append(nn.ReLU())

        for _ in range(n_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_size, 1))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)


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


class CombiDataset(Dataset):
    def __init__(self, combis, risks):
        self.combis = combis
        self.risks = risks

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


# %%
def load_dataset(dataset_path):
    dataset = pd.read_csv("../datasets/combinations/" + dataset_path + ".csv")

    with open("../datasets/patterns/" + dataset_path + ".json", "r") as f:
        patterns = json.load(f)

    features = dataset.iloc[:, :-3]
    risks = dataset.iloc[:, -3]

    n_obs, n_dim = features.shape

    return features, risks, patterns, n_obs, n_dim


def setup_data(dataset, batch_size, n_obs):
    combis, risks, _, _, n_dim = load_dataset(dataset)

    combis, risks = (
        torch.tensor(combis.values).float(),
        torch.tensor(risks.values).unsqueeze(1).float(),
    )

    X_train, y_train = combis[:n_obs], risks[:n_obs]
    X_val, y_val = combis[n_obs:], risks[n_obs:]

    training_data = CombiDataset(X_train, y_train)
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
    plt.ylim(0, 3)
    plt.xlim(0, 3)
    plt.scatter(true, pred, alpha=0.1)
    plt.plot([0, 3], [0, 3], color="black", linestyle="dashed", label="Perfection")
    plt.legend()

    return fig


def save_metrics(array, path):
    dir_ = "/".join(path.split("/")[:-1])
    os.makedirs(dir_, exist_ok=True)
    np.save(path, array)


# %% [markdown]
# # Configurations of parameters

# %%
param_values = {
    "width": [32, 64, 128],
    "hidden": [1, 2, 3, 4, 5],
    "n_obs": [100, 1000, 10000],
    "dataset": [
        "50_rx_100000_combis_4_patterns_3",
        "100_rx_100000_combis_10_patterns_35",
        "1000_rx_100000_combis_10_patterns_25",
    ],
    "decay": [0, 0.01, 0.1],
    "lr": [1e-2, 1e-1],
    "custom_layers": [False],
}

# param_values = {
#     "width": [None],
#     "hidden": [None],
#     "n_obs": [100, 1000, 10000],
#     "dataset": [
#         "50_rx_100000_combis_4_patterns_3",
#         "100_rx_100000_combis_10_patterns_35",
#         "1000_rx_100000_combis_10_patterns_25",
#     ],
#     "decay": [0],
#     "lr": [1e-2],
#     "custom_layers": [[512, 256], [512, 256, 128], [512, 256, 128, 64]],
# }

configs = [dict(zip(param_values, v)) for v in product(*param_values.values())]

# configs = [{"width": 32, "n_obs": 1000, "dataset":"50_rx_100000_combis_4_patterns_3", "decay": 0, "hidden": 3, "lr": 1e-2}, {"width": 32, "n_obs": 100, "dataset":"50_rx_100000_combis_4_patterns_3", "decay": 0, "hidden": 3, "lr": 1e-2}]
print(len(configs))
n_epochs = 25
criterion = torch.nn.MSELoss()
seeds = list(range(25))
batch_size = 32
patience = 3


# %% [markdown]
# # Train

# %% [markdown]
# ### What impacts overfitting here ?
# * Optimizer - Adam fits faster and more than SGD
# * Batch size - Lower batch size seems to lead to more overfitting. Larger ones seem to average out extremes in the input dataset during backprop
# * Net width - Obviously
# * Hidden layers - Obviously

# %%
for config in tqdm(configs):
    n_layers = config["hidden"]
    width = config["width"]
    n_obs = config["n_obs"]
    decay = config["decay"]
    dataset = config["dataset"]
    lr = config["lr"]
    custom_layers = config["custom_layers"]
    train_losses = []
    val_losses = []

    if custom_layers:
        exp_dir = f"{dataset}/{custom_layers=}_{n_obs=}_{decay=}_{lr=}"
    else:
        exp_dir = f"{dataset}/{width=}_{n_layers=}_{n_obs=}_{decay=}_{lr=}"

    print(f"doing {config}")
    # Train for 25 seeds
    for i, seed in enumerate(seeds):
        logdir = f"runs/{exp_dir}/{seed=}"
        writer = SummaryWriter(log_dir=logdir)
        min_val_loss = float("inf")

        seed_train_losses = [np.nan] * n_epochs
        seed_val_losses = [np.nan] * n_epochs
        early_stopping = EarlyStopping(patience=patience)

        make_deterministic(seed=seed)

        trainloader, X_train, y_train, X_val, y_val, n_dim = setup_data(
            dataset, batch_size, n_obs
        )
        if custom_layers:
            net = VariableNet(n_dim, custom_layers)
        else:
            net = Network(n_dim, n_layers, width).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=decay)

        ### RECORD MODEL ###
        writer.add_graph(net, X_train[0])

        for e in range(n_epochs):
            ### TRAIN ###
            for X, y in trainloader:
                optim.zero_grad()
                train_activ = net(X)
                train_loss = criterion(train_activ, y)
                train_loss.backward()
                optim.step()

            ### EVAL ###
            with torch.no_grad():
                val_activ = net(X_val)
                val_loss = criterion(val_activ, y_val).item()
                seed_val_losses[e] = val_loss

                train_activ = net(X_train)
                train_loss = criterion(train_activ, y_train).item()
                seed_train_losses[e] = train_loss

                writer.add_scalar("Loss/train", train_loss, e)
                writer.add_scalar("Loss/val", val_loss, e)
                if val_loss < min_val_loss:
                    val_activ_min_loss = val_activ.detach().clone().cpu().numpy()
                    train_activ_min_loss = train_activ.detach().clone().cpu().numpy()
                    min_val_loss = val_loss

            ### VERIFY EARLY STOP ###
            # Is weird rn but basically we just want to record the first early stop activations, but since we also want the lowest validation error's activation we can't break out yet
            if not early_stopping.early_stop:
                early_stopping(val_loss, train_activ, val_activ)
                if early_stopping.early_stop:
                    ### PLOT EARLY STOP REPRESENTATION ###
                    train_activ_graph_early_stop = (
                        early_stopping.train_activ.cpu().numpy()
                    )
                    val_activ_graph_early_stop = early_stopping.val_activ.cpu().numpy()
                    fig_pgt_train = plot_pred_vs_gt(
                        y_train.cpu().numpy(),
                        train_activ_graph_early_stop,
                        title="Prédiction par rapport à la vérité (es) (entrainement)",
                    )

                    fig_pgt_val = plot_pred_vs_gt(
                        y_val.cpu().numpy(),
                        val_activ_graph_early_stop,
                        title="Prédiction par rapport à la vérité (es) (validation)",
                    )
                    writer.add_figure("train_pred_vs_gt_es", fig_pgt_train)
                    writer.add_figure("val_pred_vs_gt_es", fig_pgt_val)
                    writer.flush()

        ### PLOT PRED VS TRUE FOR THIS SEED  (min loss) ###
        fig_pgt_train = plot_pred_vs_gt(
            y_train.cpu().numpy(),
            train_activ_min_loss,
            title="Prédiction par rapport à la vérité (min) (entrainement)",
        )

        fig_pgt_val = plot_pred_vs_gt(
            y_val.cpu().numpy(),
            val_activ_min_loss,
            title="Prédiction par rapport à la vérité (min) (validation)",
        )
        writer.add_figure("train_pred_vs_gt_min", fig_pgt_train)
        writer.add_figure("val_pred_vs_gt_min", fig_pgt_val)

        writer.flush()
        writer.close()
        plt.close("all")

        train_losses.append(seed_train_losses)
        val_losses.append(seed_val_losses)

    ### PLOT AGGREGATE DATA FOR ALL SEEDS ###
    logdir = f"runs/{exp_dir}/aggregate"
    writer = SummaryWriter(log_dir=logdir)

    ### PLOTS ###
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    fig_loss = plot_losses(n_epochs, train_losses, val_losses)
    writer.add_figure("losses", fig_loss)
    writer.flush()
    writer.close()
    plt.close("all")
    print(plt.get_fignums())

    save_metrics(train_losses, f"metrics/{exp_dir}/train_losses")
    save_metrics(val_losses, f"metrics/{exp_dir}/val_losses")
