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
import logging

logging.basicConfig(level=logging.INFO)


class Network(nn.Module):
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


class DENeuralTSDiag:
    def __init__(self, net, lamdba=1, nu=1, style="ts", sampletype="r"):
        self.net = net
        self.lamdba = lamdba
        self.total_param = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        self.len = 0
        self.U = lamdba * torch.ones((self.total_param,)).cuda()
        self.nu = nu
        self.style = style
        self.sampletype = sampletype
        self.loss_func = nn.MSELoss()
        self.vec_history = torch.tensor([]).unsqueeze(0).cuda()
        self.reward_history = torch.tensor([]).unsqueeze(0).cuda()

    def compute_activation_and_grad(self, vec):
        self.net.zero_grad()
        mu = self.net(vec)
        mu.backward()
        g_list = torch.cat(
            [p.grad.flatten().detach() for p in self.net.parameters()],
        )
        mu = mu.detach()
        return mu, g_list

    def get_sample(self, vec):
        # torch.set_grad_enabled(True)
        mu, g_list = self.compute_activation_and_grad(vec)
        cb = torch.sum(g_list * g_list / self.U)
        cb = torch.sqrt(self.lamdba * cb)

        if self.sampletype == "r":
            sigma = self.nu * cb
        elif self.sampletype == "f":
            # Exploration is generated by the dropout value, reflected in mu
            return mu.item(), g_list, mu.item(), cb.item()

        if self.style == "ts":
            sample_r = torch.normal(mu.view(-1), sigma.view(-1))
        elif self.style == "ucb":
            sample_r = mu.view(-1) + sigma.view(-1)

        # torch.set_grad_enabled(False)
        return sample_r.item(), g_list, mu.item(), cb.item()

    def train(self, n_optim_steps, lr=1e-2):

        optimizer = optim.SGD(self.net.parameters(), lr=lr)
        for _ in range(n_optim_steps):
            self.net.zero_grad()
            optimizer.zero_grad()
            pred = self.net(self.vec_history)
            loss = self.loss_func(pred, self.reward_history)
            loss.backward()
            optimizer.step()

        self.net.zero_grad()
        return loss.detach().item()

    def find_solution_in_vecs(self, vecs, thresh):
        """Find and return solutions according to the threshold in the given set of vectors
        A vector is part of the solution if its activation `mu` and its confidence interval `cb` is entirely contained above the threshold.
        i.e. mu - cb > thresh -> vector is in solution
        Args:
            vecs (torch.Tensor/list of torch.Tensor): List of vectors to check for solution membership
            thresh (float): Threshold against which to compare for the solution
        Returns:
            tensor: torch.Tensor of torch.Tensor of the solution
        """

        solution = []
        for vec in vecs:
            mu, g_list = self.compute_activation_and_grad(vec)
            cb = torch.sum(g_list * g_list / self.U)
            cb = torch.sqrt(self.lamdba * cb)
            logging.info(f"{mu-cb}, {cb} : {thresh}")
            if (mu - cb).item() > thresh:
                solution.append(vec)

        if solution:
            solution = torch.stack(solution)
        else:
            solution = torch.tensor([])

        return solution


class LenientDENeuralTSDiag(DENeuralTSDiag):
    def __init__(self, net, reward_sample_thresholds, lamdba=1, nu=1, sampletype="r"):
        self.net = net
        self.reward_sample_thresholds = reward_sample_thresholds
        self.lamdba = lamdba
        self.total_param = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        # self.len = 0
        self.U = lamdba * torch.ones((self.total_param,)).cuda()
        self.nu = nu
        self.sampletype = sampletype
        self.loss_func = nn.MSELoss()
        self.vec_history = torch.tensor([]).unsqueeze(0).cuda()
        self.reward_history = torch.tensor([]).unsqueeze(0).cuda()

    def get_sample(self, vec):
        mu, g_list = self.compute_activation_and_grad(vec)

        cb = torch.sum(g_list * g_list / self.U)
        cb = torch.sqrt(self.lamdba * cb)

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
        return sample_r.item(), g_list, mu.item(), cb.item()
