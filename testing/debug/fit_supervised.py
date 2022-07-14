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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.tensorboard import SummaryWriter

sys.path.append("../..")
sys.path.append("..")
from utils import *
from fit_sup_utils import *

from tqdm import tqdm


matplotlib.use("Agg")

plt.rcParams["figure.figsize"] = (7, 5)  # default = (6.4, 4.8)
# plt.rcParams["text.usetex"] = True
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
# param_values = {
#     "dataset": [
#         "50_rx_100000_combis_4_patterns_3",
#         "100_rx_100000_combis_10_patterns_35",
#         "1000_rx_100000_combis_10_patterns_25",
#     ],
#     "width": [False],
#     "hidden": [False],
#     "n_obs": [20000],
#     "decay": [0],
#     "lr": [0.001],
#     "custom_layers": [[512, 256, 128], [128, 64, 32], [64, 32]],
#     "reweight": [None, True, "sqrt_inv"],
#     "batch_size": [32],
#     "dropout_rate" : [None],
#     "loss": [["mse"], ["rmse"]]
# }

param_values = {
    "dataset": [
        # "50_rx_100000_combis_4_patterns_3",
        # "100_rx_100000_combis_10_patterns_35",
        "1000_rx_100000_combis_10_patterns_25",
    ],
    "width": [128],
    "hidden": [3],
    "n_obs": [20000],
    "decay": [0],
    "lr": [0.001],
    "custom_layers": [None],
    "reweight": ["sqrt_inv"],
    "batch_size": [32],
    "dropout_rate": [None],
    "loss": [["quantile", [0.5, 0.7]]],
    "classif_thresh": [None],
    "batch_norm": [True],
    "patience": [5, 10],
    "validation": [None],
}

configs = [dict(zip(param_values, v)) for v in product(*param_values.values())]
print(len(configs))

# %% [markdown]
# # Train

# %% [markdown]
# ## What affects overfitting here ?
# * Optimizer - Adam fits faster and more than SGD
# * Learning rate - Lower (0.001) seems better than default (1e-2)
# * Batch size - Lower batch size seems to lead to more overfitting. Larger ones seem to average out extremes in the input dataset during backprop
# * Net width - Obviously
# * Hidden layers - Obviously

# %% [markdown]
# ## What helps fitting "outliers" ?
# Fitting is meant in a broad sense here and just means "Not predicting the mean"
#
# * Lower LR (?) 0.001 is better than 0.01
# * Label dist. smoothing, but seems to affect validation perf for the "common" cluster (not outlier)
# * Quantile loss seems to help over estimating values properly when using a quantile at 0.75 (i.e. low risk cluster is overestimated to a lesser extent than high risk cluster, which could be good for the bandit algorithm), HOWEVER, fitting a quantile that isn't 50 feels like it could mess up with NeuralTS (since we're not predicting a mean anymore, we're predicting a quantile, and the output of the NN usually goes into a Normal distribution as the mean param)
# * Label dist smoothing with a bigger batch size seems to help

# %% [markdown]
# ### What does not work
#
# * l1 loss
# * High LR (0.01) seems to lead to high bias in validation

# %% [markdown]
# ### What does work
# * 3 layers of 128 width seem to be sufficient to overfit in training
# * lr of 0.001 seem to help that overfitting, nice, lr of 0.01 seems to fail overfitting in some cases (dataset 100 and 50)
# * mse is better than rmse
# * If doing LDS, sqrt_inv is better than just True
# * for 1000 rx, small batches, sqrt_inv LDS, smaller models seems to have less bias in validation

# %% [markdown]
# ### To try
# * ~~Very simple network with LDS (like one that can hardly overfit)~~
# * Transform regression into a classification with the new knowledge (how to interpret CIs then ?)
# * ~~High batch size with sqrt_inv LDS and lower LR~~
# * ~~Adaptive LR (Plateau)~~
# * ~~Embed combination vectors. Each RX is a word, each combination of Rx is like a sentence. Embedding should pick up a relationship between the Rxs~~

# %% [markdown]
# ### Goal
# * Generalize in validation
# * If unable to generalize in the conventional sense, at least make sure the "high risk" cluster is estimated over the "low risk" cluster

# %%
n_epochs = 100
# seeds = list(range(25))
seeds = [0]

# %%
def run_config(config, exp_dir="test_graal"):
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
    loss_info = config["loss"]
    classif_thresh = config["classif_thresh"]
    batch_norm = config["batch_norm"]
    patience = config["patience"]
    validation = config["validation"]

    n_outputs = 1
    pred_idx = 0

    criterion = get_loss(*loss_info)
    if loss_info[0] == "quantile":
        quantiles = np.array(loss_info[1])
        assert 0.5 in quantiles
        pred_idx = np.where(quantiles == 0.5)[0][0]
        n_outputs = len(quantiles)

    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []
    if exp_dir == None:
        exp_dir == ""
    l = []
    for k, v in config.items():
        l += [f"{k}={v}"]

    exp_dir += "/" + "/".join(l)

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

        trainloader, training_data, X_val, y_val, n_dim = setup_data(
            dataset, batch_size, n_obs, reweight, classif_thresh, validation
        )

        X_train, y_train = training_data.combis, training_data.labels
        if custom_layers is not None:
            net = VariableNet(n_dim, custom_layers)
        else:
            net = Network(
                n_dim, n_layers, n_outputs, width, dropout_rate, batch_norm
            ).to(device)

        if lr == "plateau":
            optim = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=decay)
            sched = ReduceLROnPlateau(optim, "min", patience=patience - 2)
        else:
            optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=decay)
        ### RECORD MODEL ###
        writer.add_graph(net, X_train)

        for e in range(n_epochs):
            ### TRAIN ###
            for X, y in trainloader:
                optim.zero_grad()
                train_activ = net(X)
                train_loss = criterion(train_activ, y)
                if loss_info[0] == "rmse":
                    train_loss = torch.sqrt(train_loss)
                train_loss.backward()
                optim.step()

            ### EVAL ###
            with torch.no_grad():
                # Compute losses
                val_activ = net(X_val)
                val_loss = criterion(val_activ, y_val)

                train_activ = net(X_train)
                train_loss = criterion(train_activ, y_train)

                if loss_info[0] == "rmse":
                    train_loss = torch.sqrt(train_loss)
                    val_loss = torch.sqrt(val_loss)

                val_loss = val_loss.item()
                train_loss = train_loss.item()

                # Get R2 metric
                train_r2 = r2_score(
                    y_train.cpu().numpy(), train_activ.cpu().numpy()[:, pred_idx]
                )
                val_r2 = r2_score(
                    y_val.cpu().numpy(), val_activ.cpu().numpy()[:, pred_idx]
                )

                # Save
                seed_train_losses[e] = train_loss
                seed_val_losses[e] = val_loss
                seed_train_r2s[e] = train_r2
                seed_val_r2s[e] = val_r2

                writer.add_scalar("Loss/train", train_loss, e)
                writer.add_scalar("Loss/val", val_loss, e)
                writer.add_scalar("R2/train", train_r2, e)
                writer.add_scalar("R2/val", val_r2, e)

                # Update LR scheduler
                if type(lr) == str:
                    sched.step(val_loss)

                # Update minimums
                if val_loss < min_val_loss:
                    val_activ_min_loss = (
                        val_activ.detach().clone().cpu().numpy()[:, pred_idx]
                    )
                    train_activ_min_loss = (
                        train_activ.detach().clone().cpu().numpy()[:, pred_idx]
                    )
                    min_val_loss = val_loss
                if train_loss < min_train_loss:
                    val_activ_mintrain_loss = (
                        val_activ.detach().clone().cpu().numpy()[:, pred_idx]
                    )
                    train_activ_mintrain_loss = (
                        train_activ.detach().clone().cpu().numpy()
                    )[:, pred_idx]
                    min_train_loss = train_loss

            ### VERIFY EARLY STOP ###
            # Is weird rn but basically we just want to record the first early stop activations, but since we also want the lowest validation error's activation we can't break out yet
            if not early_stopping.early_stop:
                early_stopping(val_loss, train_activ, val_activ)
                if early_stopping.early_stop:
                    ### PLOT EARLY STOP REPRESENTATION ###
                    train_activ_graph_early_stop = (
                        early_stopping.train_activ.cpu().numpy()
                    )[:, pred_idx]
                    val_activ_graph_early_stop = early_stopping.val_activ.cpu().numpy()[
                        :, pred_idx
                    ]

                    fig_pgt_es = plot_pred_vs_gt(
                        y_train.cpu().numpy(),
                        train_activ_graph_early_stop,
                        y_val.cpu().numpy(),
                        val_activ_graph_early_stop,
                        title="Prédiction par rapport à la vérité (Early Stop)",
                    )

                    writer.add_figure("pred_vs_gt_early_stop", fig_pgt_es)
                    writer.flush()

        ### PLOT PRED VS TRUE FOR THIS SEED  (min val loss) ###
        fig_pgt_minval = plot_pred_vs_gt(
            y_train.cpu().numpy(),
            train_activ_min_loss,
            y_val.cpu().numpy(),
            val_activ_min_loss,
            title="Prédiction par rapport à la vérité (min val)",
        )

        writer.add_figure("pred_vs_gt_minval", fig_pgt_minval)

        ### PLOT PRED VS TRUE FOR THIS SEED  (min train loss) ###
        fig_pgt_mintrain = plot_pred_vs_gt(
            y_train.cpu().numpy(),
            train_activ_mintrain_loss,
            y_val.cpu().numpy(),
            val_activ_mintrain_loss,
            title="Prédiction par rapport à la vérité (min train)",
        )

        writer.add_figure("pred_vs_gt_mintrain", fig_pgt_mintrain)
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

    # train_r2s = np.array(train_r2s)
    # val_r2s = np.array(val_r2s)
    # fig_r2 = plot_losses(n_epochs, train_r2s, val_r2s)

    # writer.add_figure("losses", fig_loss)
    # writer.add_figure("r2s", fig_r2)

    # writer.flush()
    # writer.close()
    plt.close("all")

    save_metrics(train_losses, f"metrics/{exp_dir}/train_losses")
    save_metrics(val_losses, f"metrics/{exp_dir}/val_losses")
    save_metrics(train_r2s, f"metrics/{exp_dir}/train_r2s")
    save_metrics(val_r2s, f"metrics/{exp_dir}/val_r2s")

    # print(f"saved at runs/{exp_dir}")


# %%
with Pool(4) as p:
    p.map(run_config, configs)

# %%
