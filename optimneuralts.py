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
    def __init__(self, dim, hidden_size=100, mask=None, dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.mask = mask
        self.hidden_size = hidden_size
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        return self.fc2(self.activate(self.dropout(self.fc1(x))))


class ReplayDataset(Dataset):
    def __init__(self, features=None, rewards=None):
        self.features = features
        self.rewards = rewards

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        return self.features[idx], self.rewards[idx]

    def set_(self, features_2d, rewards_2d):
        self.features = features_2d
        self.rewards = rewards_2d

    def add(self, features, reward):
        if features is None or reward is None:
            return
        self.features = torch.cat((self.features, features))
        self.rewards = torch.cat((self.rewards, reward))


class ValidationReplayDataset(ReplayDataset):
    def __init__(self, features=None, rewards=None, valtype=None):
        self.features = None
        self.rewards = None
        self.valtype = valtype
        self.bins_features = {}
        self.bins_rewards = {}

        if features is None or rewards is None:
            pass
        else:
            self.set_(features, rewards)

    def build_mapping(self):
        for i in range(len(self.rewards)):
            reward = self.rewards[i].unsqueeze(0)
            vec = self.features[i].unsqueeze(0)

            bin_ = int(reward * 10) / 10

            self.bins_features[bin_] = vec
            self.bins_rewards[bin_] = reward

    def __len__(self):
        return len(self.rewards)

    def set_(self, features_2d, rewards_2d):
        self.features = features_2d
        self.rewards = rewards_2d

        if self.valtype == "bins":
            self.build_mapping()

    def update(self, features, reward):
        to_training = (features, reward)
        if self.valtype == "extrema":
            # If valtype is extrema, then the set is composed of only 2 observations, the min and max of rewards
            # Assumes sorted tensors based on self.rewards
            if reward < min(self.rewards):
                to_training = (
                    self.features[0].clone().unsqueeze(0),
                    self.rewards[0].clone().unsqueeze(0),
                )
                self.features[0] = features[0]
                self.rewards[0] = reward[0]
            elif reward > max(self.rewards):
                to_training = (
                    self.features[1].clone().unsqueeze(0),
                    self.rewards[1].clone().unsqueeze(0),
                )
                self.features[1] = features[0]
                self.rewards[1] = reward[0]

        elif self.valtype == "bins":
            # TODO TEST THIS
            # Update the bins with the new observation
            new_obs_bin = int(reward * 10) / 10
            # Send the observation that was already in the bin to the training set
            to_training = (
                self.bins_features.get(new_obs_bin, None),
                self.bins_rewards.get(new_obs_bin, None),
            )
            # Update the bin
            self.bins_features[new_obs_bin] = features
            self.bins_rewards[new_obs_bin] = reward

            self.rewards = torch.cat(list(self.bins_rewards.values()))
            self.features = torch.cat(list(self.bins_features.values()))

        return to_training


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
        self.best_model = None

    def __call__(self, cur_loss, model):
        # If no improvement
        if cur_loss >= self.min_loss:
            self.count += 1
        else:  # Improvement, store state dict
            self.count = 0
            self.store(model)
            self.min_loss = cur_loss

    def store(self, model):
        self.best_model = deepcopy(model)
        self.best_model.zero_grad()

    @property
    def early_stop(self):
        if self.count >= self.patience:
            return True


class Network(nn.Module):
    def __init__(self, dim, n_hidden_layers, hidden_size=100):
        super().__init__()
        self.model = nn.Sequential()

        self.model.add_module("Linear 0", nn.Linear(dim, hidden_size))
        self.model.add_module("ReLU 0", nn.ReLU())

        for i in range(n_hidden_layers - 1):
            self.model.add_module(
                f"Linear {i + 1}", nn.Linear(hidden_size, hidden_size)
            )
            self.model.add_module(f"ReLU {i + 1}", nn.ReLU())

        self.model.add_module(f"Linear Final", nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.model(x)


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
        valtype="extrema",
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
        self.train_dataset = ReplayDataset()
        self.val_dataset = ValidationReplayDataset(valtype=valtype)
        optimizers = {"sgd": optim.SGD, "adam": optim.Adam}
        # Keep optimizer separate from DENeuralTS class to tune lr as we go through timesteps if we so desire
        self.optimizer_class = optimizers[optim_string]

    def compute_activation_and_grad(self, vec):
        self.net.zero_grad()
        mu = self.net(vec)
        mu.backward(retain_graph=True)
        g_list = torch.cat([p.grad.flatten() for p in self.net.parameters()])
        return mu, g_list

    def get_sample(self, vec):
        mu, g_list = self.compute_activation_and_grad(vec)
        sigma = torch.tensor(0.0)
        sigma = sigma + torch.sum(g_list * g_list / self.U)
        sigma = torch.sqrt(self.lambda_ * sigma)

        if self.sampletype == "f":
            # Exploration is generated by the dropout value, reflected in mu
            mu = mu.detach()
            return mu, g_list, mu.item(), sigma.detach().item()

        if self.style == "ts":
            sample_r = torch.distributions.Normal(
                mu.view(-1), self.nu * sigma.view(-1)
            ).rsample()
        elif self.style == "ucb":
            sample_r = mu.view(-1) + sigma.view(-1)

        return sample_r, g_list, mu.detach().item(), sigma.detach().item()

    def train(
        self,
        n_epochs,
        lr=1e-2,
        batch_size=-1,
        generator=torch.Generator(device="cuda"),
        patience=5,
    ):
        # For full batch grad descent
        if batch_size == -1:
            batch_size = len(self.train_dataset)

        # Setup
        self.len += 1
        weight_decay = self.decay * (self.lambda_ / self.len)
        optimizer = self.optimizer_class(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Although we're giving the whole dataset, the class is made such that only training examples are used here
        loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        early_stop = EarlyStopping(patience)
        # Train loop
        for _ in range(n_epochs):
            for X, y in loader:
                optimizer.zero_grad()
                pred = self.net(X)
                loss = self.loss_func(pred, y)
                loss.backward()
                optimizer.step()

            val_loss = get_validation_loss(self.net, self.val_dataset, self.loss_func)

            early_stop(val_loss, self.net)

            if early_stop.early_stop:
                break

        optimizer.zero_grad()

        # Set the net to the best in validation we've seen
        # Deep copy to ensure no remaining references to the early_stop object
        self.net = deepcopy(early_stop.best_model)
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


def get_validation_loss(net, val_dataset, loss_fn):
    X_val, y_val = val_dataset.features, val_dataset.rewards
    with torch.no_grad():
        pred = net(X_val)
        loss = loss_fn(pred, y_val)
    return loss
