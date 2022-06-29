import logging
import time
import types
from copy import deepcopy
from math import sqrt
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from detorch import DE, Policy, Strategy
from detorch.config import Config, default_config
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(level=logging.INFO)


class NetworkDropout(nn.Module):
    def __init__(self, dim, hidden_size=100, mask=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.mask = mask
        self.hidden_size = hidden_size

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        if self.mask is None:
            return self.fc2(self.activate(self.fc1(x)))
        else:
            return self.fc2(self.activate(self.mask * self.fc1(x)))


class ReplayDataset(Dataset):
    def __init__(self, features=None, rewards=None):
        self.hist_features = features
        self.hist_rewards = rewards
        self.train_features = None
        self.train_rewards = None
        self.val_features = None
        self.val_rewards = None

    def __len__(self):
        return len(self.hist_rewards)

    def __getitem__(self, idx):
        return self.hist_features[idx], self.hist_rewards[idx]

    def set_hists(self, hist_features, hist_rewards):
        self.hist_features = hist_features
        self.hist_rewards = hist_rewards

    def add(self, features, reward):
        self.hist_features = torch.cat((self.hist_features, features))
        self.hist_rewards = torch.cat((self.hist_rewards, reward))

    def update_validation_set(self, split_size=0.1):

        return


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


class Network(nn.Module):
    def __init__(self, dim, n_hidden_layers, hidden_size=100):
        super().__init__()
        self.activation = nn.ReLU()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(dim, hidden_size))
        for _ in range(n_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, 1))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class DENeuralTSDiag:
    def __init__(
        self,
        net,
        optim_string="sgd",
        lambda_=1,
        nu=1,
        style="ts",
        sampletype="r",
        decay=False,
    ):
        self.net = net
        self.lambda_ = lambda_
        self.total_param = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        self.len = 0
        self.U = lambda_ * torch.ones((self.total_param,)).cuda()
        self.nu = nu
        self.style = style
        self.sampletype = sampletype
        self.decay = decay

        self.loss_func = nn.MSELoss()
        # self.vec_history = torch.tensor([]).unsqueeze(0).cuda()
        # self.reward_history = torch.tensor([]).unsqueeze(0).cuda()
        self.dataset = ReplayDataset()
        optimizers = {"sgd": optim.SGD, "adam": optim.Adam}
        # Keep optimizer separate from DENeuralTS class to tune lr as we go through timesteps if we so desire
        self.optimizer_class = optimizers[optim_string]

    def compute_activation_and_grad(self, vec):
        self.net.zero_grad()
        mu = self.net(vec)
        mu.backward(retain_graph=True)
        g_list = torch.cat(
            [p.grad.flatten().detach() for p in self.net.parameters()],
        )
        # mu = mu.detach()
        return mu, g_list

    def get_sample(self, vec):
        mu, g_list = self.compute_activation_and_grad(vec)
        cb = torch.sum(g_list * g_list / self.U)
        cb = torch.sqrt(self.lambda_ * cb)

        if self.sampletype == "r":
            sigma = self.nu * cb
        elif self.sampletype == "f":
            # Exploration is generated by the dropout value, reflected in mu
            mu = mu.detach()
            return mu, g_list, mu.item(), cb.item()

        if self.style == "ts":
            sample_r = torch.normal(mu.view(-1), sigma.view(-1))
        elif self.style == "ucb":
            sample_r = mu.view(-1) + sigma.view(-1)

        return sample_r, g_list, mu.detach().item(), cb.item()

    def train(self, n_epochs, lr=1e-2, batch_size=32):
        # For full batch grad descent
        if batch_size == -1:
            batch_size = len(self.dataset)

        # Setup
        self.len += 1
        weight_decay = self.decay * (self.lambda_ / self.len)
        optimizer = self.optimizer_class(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )

        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator(device="cuda"),
        )

        # Train loop
        for _ in range(n_epochs):
            for X, y in loader:
                optimizer.zero_grad()
                pred = self.net(X)
                loss = self.loss_func(pred, y)
                loss.backward()
                optimizer.step()

        optimizer.zero_grad()

        return loss.detach().item()

    def find_solution_in_vecs(self, vecs, thresh, n_sigmas):
        """Find and return solutions according to the threshold in the given set of vectors
        A vector is part of the solution if its activation `mu` and 3 times its std is entirely contained above the threshold.
        i.e. mu - 3 * sigma > thresh -> vector is in solution
        Args:
            vecs (torch.Tensor/list of torch.Tensor): List of vectors to check for solution membership
            thresh (float): Threshold against which to compare for the solution
        Returns:
            tensor: torch.Tensor of torch.Tensor of the solution
        """

        solution = []
        mus = []
        sigmas = []
        for vec in vecs:
            mu, g_list = self.compute_activation_and_grad(vec)
            sigma = torch.sum(g_list * g_list / self.U)
            sigma = torch.sqrt(self.lambda_ * sigma)

            if (mu - n_sigmas * sigma).item() > thresh:
                solution.append(vec)

            mus.append(mu.item())
            sigmas.append(3 * sigma.item())
        if solution:
            solution = torch.stack(solution)
        else:
            solution = torch.tensor([])

        return (solution, mus, sigmas)


class LenientDENeuralTSDiag(DENeuralTSDiag):
    def __init__(self, reward_sample_thresholds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_sample_thresholds = reward_sample_thresholds

    def get_sample(self, vec):
        mu, g_list = self.compute_activation_and_grad(vec)

        cb = torch.sum(g_list * g_list / self.U)
        cb = torch.sqrt(self.lambda_ * cb)

        sigma = self.nu * cb

        # Make sure sample is within the sampling thresholds for the expected reward
        if mu.item() > self.reward_sample_thresholds[1]:
            sample_r = mu
        else:
            sample_r = torch.cuda.FloatTensor([0])
            torch.nn.init.trunc_normal_(
                sample_r, mu.view(-1), sigma.view(-1), *self.reward_sample_thresholds
            )
        # torch.set_grad_enabled(False)
        return sample_r, g_list, mu.detach().item(), cb.item()
