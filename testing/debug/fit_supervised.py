# %%
import json
import logging
import sys
import os
import random
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from math import floor, ceil
from torch.nn.functional import conv1d
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../..")
sys.path.append("..")
from utils import *
from fit_sup_utils import *

# from tqdm import tqdm


matplotlib.use("Agg")

plt.rcParams["figure.figsize"] = (7, 5)  # default = (6.4, 4.8)
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 140  # default = 100
plt.rcParams["font.family"] = "serif"
# plt.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
plt.style.use("ggplot")
title_font_size = "10"


# %% [markdown]
# # Globals

# %%
THRESH = 1.1
device = torch.device("cuda")


# %% [markdown]
# # Generate configurations

# %%
param_values = {
    "dataset": [
        "50_rx_100000_combis_4_patterns_3",
        "100_rx_100000_combis_10_patterns_35",
        "1000_rx_100000_combis_10_patterns_25",
    ],
    "width": [64, 128, 256],
    "hidden": [3, 4, 5],
    "n_obs": [20000],
    "decay": [0],
    "lr": [1e-2],
    "custom_layers": [None],
    "reweight": [None, "sqrt_inv"],
    "batch_size": [32],
    "dropout_rate": [None],
    "loss": ["rmse", "mse"],
}

configs = [dict(zip(param_values, v)) for v in product(*param_values.values())]

print(len(configs))

# %% [markdown]
# # Train

# %% [markdown]
# ## What affects overfitting here ?
# * Optimizer - Adam fits faster and more than SGD
# * Batch size - Lower batch size seems to lead to more overfitting. Larger ones seem to average out extremes in the input dataset during backprop
# * Net width - Obviously
# * Hidden layers - Obviously

# %%
n_epochs = 100
# seeds = list(range(25))
seeds = [0]
patience = 5

# %%


# %%
def run_config(config, exp_dir="test_r2", modifier="l1loss"):
    n_layers = config["hidden"]
    width = config["width"]
    n_obs = config["n_obs"]
    decay = config["decay"]
    dataset = config["dataset"]
    lr = config["lr"]
    custom_layers = config["custom_layers"]
    reweight = config["reweight"]
    batch_size = config["batch_size"]
    dropout_rate = config["dropout_rate"]
    loss_name = config["loss"]

    criterion = get_loss(loss_name)

    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []
    if exp_dir == None:
        exp_dir == ""
    l = []
    for k, v in config.items():
        l += [f"{k}={v}"]

    exp_dir += "/" + "-".join(l) + f"-{modifier=}"

    # Train for 25 seeds
    for i, seed in enumerate(seeds):
        logdir = f"runs/{exp_dir}/{seed=}"
        writer = SummaryWriter(log_dir=logdir)
        min_val_loss = float("inf")
        min_train_loss = float("inf")

        seed_train_losses = [np.nan] * n_epochs
        seed_val_losses = [np.nan] * n_epochs
        seed_train_r2s = [np.nan] * n_epochs
        seed_val_r2s = [np.nan] * n_epochs
        early_stopping = EarlyStoppingActiv(patience=patience)

        make_deterministic(seed=seed)

        trainloader, X_train, y_train, X_val, y_val, n_dim = setup_data(
            dataset, batch_size, n_obs, reweight
        )
        if custom_layers is not None:
            net = VariableNet(n_dim, custom_layers)
        else:
            net = Network(n_dim, n_layers, width, dropout_rate).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=decay)

        ### RECORD MODEL ###
        writer.add_graph(net, X_train[0])

        for e in range(n_epochs):
            ### TRAIN ###
            for X, y in trainloader:
                optim.zero_grad()
                train_activ = net(X)
                train_loss = criterion(train_activ, y)
                if loss_name == "rmse":
                    train_loss = torch.sqrt(train_loss)
                train_loss.backward()
                optim.step()

            ### EVAL ###
            with torch.no_grad():
                # Compute losses
                val_activ = net(X_val)
                val_loss = criterion(val_activ, y_val)

                if loss_name == "rmse":
                    val_loss = torch.sqrt(val_loss)

                train_activ = net(X_train)
                train_loss = criterion(train_activ, y_train)

                if loss_name == "rmse":
                    train_loss = torch.sqrt(train_loss)

                val_loss = val_loss.item()
                train_loss = train_loss.item()

                # Get R2 metric
                train_r2 = r2_score(y_train.cpu().numpy(), train_activ.cpu().numpy())
                val_r2 = r2_score(y_val.cpu().numpy(), val_activ.cpu().numpy())

                # Save
                seed_train_losses[e] = train_loss
                seed_val_losses[e] = val_loss
                seed_train_r2s[e] = train_r2
                seed_val_r2s[e] = val_r2

                writer.add_scalar("Loss/train", train_loss, e)
                writer.add_scalar("Loss/val", val_loss, e)
                writer.add_scalar("R2/train", train_r2, e)
                writer.add_scalar("R2/val", val_r2, e)

                # Update minimums
                if val_loss < min_val_loss:
                    val_activ_min_loss = val_activ.detach().clone().cpu().numpy()
                    train_activ_min_loss = train_activ.detach().clone().cpu().numpy()
                    min_val_loss = val_loss
                if train_loss < min_train_loss:
                    val_activ_mintrain_loss = val_activ.detach().clone().cpu().numpy()
                    train_activ_mintrain_loss = (
                        train_activ.detach().clone().cpu().numpy()
                    )
                    min_train_loss = train_loss

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
            title="Prédiction par rapport à la vérité (minval) (entrainement)",
        )

        fig_pgt_val = plot_pred_vs_gt(
            y_val.cpu().numpy(),
            val_activ_min_loss,
            title="Prédiction par rapport à la vérité (minval) (validation)",
        )

        writer.add_figure("train_pred_vs_gt_minval", fig_pgt_train)
        writer.add_figure("val_pred_vs_gt_minval", fig_pgt_val)
        ### PLOT PRED VS TRUE FOR THIS SEED  (min loss) ###
        fig_pgt_train_mintrain = plot_pred_vs_gt(
            y_train.cpu().numpy(),
            train_activ_mintrain_loss,
            title="Prédiction par rapport à la vérité (mintrain) (entrainement)",
        )

        fig_pgt_val_mintrain = plot_pred_vs_gt(
            y_val.cpu().numpy(),
            val_activ_mintrain_loss,
            title="Prédiction par rapport à la vérité (mintrain) (validation)",
        )

        print(val_activ_min_loss.min())
        print(val_activ_min_loss.max())
        writer.add_figure("train_pred_vs_gt_mintrain", fig_pgt_train_mintrain)
        writer.add_figure("val_pred_vs_gt_mintrain", fig_pgt_val_mintrain)
        writer.flush()
        writer.close()
        plt.close("all")

        train_losses.append(seed_train_losses)
        val_losses.append(seed_val_losses)
        train_r2s.append(seed_train_r2s)
        val_r2s.append(seed_val_r2s)
    ### PLOT AGGREGATE DATA FOR ALL SEEDS ###
    # logdir = f"runs/{exp_dir}/aggregate"
    # writer = SummaryWriter(log_dir=logdir)

    # ### PLOTS ###
    # train_losses = np.array(train_losses)
    # val_losses = np.array(val_losses)
    # fig_loss = plot_losses(n_epochs, train_losses, val_losses)
    # writer.add_figure("losses", fig_loss)
    # writer.flush()
    # writer.close()
    plt.close("all")

    save_metrics(train_losses, f"metrics/{exp_dir}/train_losses")
    save_metrics(val_losses, f"metrics/{exp_dir}/val_losses")
    save_metrics(train_r2s, f"metrics/{exp_dir}/train_r2s")
    save_metrics(val_r2s, f"metrics/{exp_dir}/val_r2s")

    print(f"saved at runs/{exp_dir}")


# %%
for config in configs:
    run_config(config)
