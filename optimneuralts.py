import time
import types
from copy import deepcopy
from math import sqrt
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from detorch import DE, Policy, Strategy
from detorch.config import Config, default_config
from torch import optim
import logging

logging.basicConfig(level=logging.INFO)

# from torch_utils import torch_argmax_best_value, torch_argmin_best_value
# torch.set_default_dtype(torch.float64)


# class NeuralUCBNet(nn.Module):
#     def __init__(self, in_dim, width, num_layers, init="zhou"):
#         """Init method for the network
#         Args:
#             in_dim (int): dimension of input (dimension of feature vectors, or contexts)
#             width (int): width of each layer (called m in Zhou et al. 2020)
#             num_layers (int): specifies the number of layers in the network - affects
#             depth
#         """
#         super().__init__()
#         assert num_layers >= 2
#         # Seems necessary for the initialization of weights via Zhou et al.'s procedure
#         assert width % 2 == 0

#         self.in_dim = in_dim
#         self.width = width
#         self.num_layers = num_layers

#         self.layers = nn.ModuleList()
#         # 1st layer is m x d
#         self.layers.append(nn.Linear(in_dim, width))
#         # 2nd to L-1-th layers are m x m
#         for _ in range(num_layers - 1):
#             self.layers.append(nn.Linear(width, width))

#         # Output layer
#         self.layers.append(nn.Linear(width, 1))

#         self.activation = nn.ReLU()
#         self.sqrt_width = sqrt(self.width)
#         if init == "zhou":
#             self.initialize_weights()

#     def initialize_weights(self):
#         """Init the net's weights according to Zhou et al.'s procedure"""
#         # Init layers 1  to L-1 according to Zhou et al. 2020.
#         std = 4 / self.width
#         half_width = int(self.width / 2)
#         W = torch.FloatTensor(half_width, self.in_dim - 1).normal_(mean=0, std=std)
#         W.requires_grad = True

#         self.layers[0].weight.data[:half_width, : self.in_dim - 1] = W
#         self.layers[0].weight.data[half_width:, 1:] = W
#         self.layers[0].weight.data[:half_width, -1].fill_(0)
#         self.layers[0].weight.data[half_width:, 0].fill_(0)

#         for layer in self.layers[1:-1]:
#             W = torch.empty((half_width, self.width - 1)).normal_(mean=0, std=std)
#             W.requires_grad = True
#             layer.weight.data[:half_width, : self.width - 1] = W
#             layer.weight.data[half_width:, 1:] = W
#             layer.weight.data[:half_width, -1].fill_(0)
#             layer.weight.data[half_width:, 0].fill_(0)

#         # Init last layer
#         std = 2 / self.width
#         w = torch.empty((half_width, 1)).normal_(mean=0, std=std)
#         w.requires_grad = True
#         self.layers[-1].weight.data[:, :half_width] = w.T
#         self.layers[-1].weight.data[:, half_width:] = -w.T

#     def forward(self, x):
#         for layer in self.layers[:-1]:
#             x = self.activation(layer(x))
#         x = self.layers[-1](x)
#         return x


# class DENeuralUCB:
#     def __init__(
#         self,
#         net,
#         optim,
#         reg,
#         exploration_mult,
#         reward_fn,
#         de_config,
#         de_policy,
#         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#     ):
#         self.device = device
#         self.net = net.to(self.device)
#         self.optim = optim
#         self.reg = reg  # lambda
#         self.exploration_mult = exploration_mult  # nu
#         self.reward_fn = reward_fn
#         self.de_config = de_config
#         self.de_policy = de_policy

#         self.net_initial_params = self.get_net_params()
#         self.net_init_state_dict = deepcopy(self.net.state_dict())

#         self.total_param = sum(
#             p.numel() for p in self.net.parameters() if p.requires_grad
#         )

#         self.Z = reg * torch.ones((self.total_param,)).cuda()
#         # self.Z = reg * torch.eye(self.p).to(self.device)
#         # self.inv_Z = torch.inverse(self.Z)
#         self.vec_history = torch.tensor([[]]).to(device)
#         self.reward_history = torch.tensor([[]]).to(device)

#     def get_net_params(self):
#         net_params = torch.Tensor([]).to(self.device)
#         for layer in self.net.layers:
#             net_params = torch.cat((net_params, layer.weight.data.flatten()), 0)
#         return net_params

#     def compute_nn_loss(self, vecs, targets):
#         # Forward pass
#         predictions = self.net(vecs)
#         current_params = self.get_net_params()
#         # Compute loss
#         mse_loss = F.mse_loss(predictions, targets, reduction="sum") * 0.5
#         current_loss = mse_loss
#         print(f"current: {current_loss / len(vecs)}")

#         return current_loss

#     def train_nn(self, batch_size, n_grad_steps):
#         """Update NN parameters
#         Args:
#             vecs (tensor): history of past played vectors
#             targets (tensor): history of past observer rewards
#             batch_size (int): batch size for the loss computation
#             n_optim_steps (int): number of optimization steps to do
#         """
#         vecs = self.vec_history
#         targets = self.reward_history
#         # Reset the parameters every time according to Zhou et al.
#         # Compute the loss
#         for i in range(n_grad_steps):
#             # Select a batch to train on with SGD
#             perm = torch.randperm(vecs.size(0))
#             idx = perm[:batch_size]
#             batch_vecs = vecs[idx]
#             batch_targets = targets[idx]
#             iter_losses = []
#             # for vec, r in zip(batch_vecs, batch_targets):
#             self.optim.zero_grad()
#             #     pred = self.net(vec)
#             #     delta = pred - r
#             #     current_loss = delta * delta
#             #     iter_losses.append(current_loss.item())
#             current_loss = self.compute_nn_loss(batch_vecs, batch_targets)
#             # Back prop
#             current_loss.backward()
#             self.optim.step()
#             # print(sum(iter_losses) / len(iter_losses))


#     def compute_activation_and_gradient(self, vec):
#         # Compute gradient on activations (as in Zhou et al. 2020)
#         self.net.zero_grad()
#         activation = self.net(vec)
#         activation.backward()
#         # Get gradient for this arm pull
#         grads = torch.cat([p.grad.flatten().detach() for p in self.net.parameters()])

#         return activation.detach(), grads

#     def compute_ucb(self, vec):
#         activation, activation_grads = self.compute_activation_and_gradient(vec)

#         exploration_bonus = (
#             self.reg
#             * self.exploration_mult
#             * activation_grads
#             * activation_grads
#             / self.Z
#         )
#         exploration_bonus = sqrt(exploration_bonus.sum())

#         return activation + self.exploration_mult * exploration_bonus, activation_grads

#     def de_best_member(self):
#         # Look for lower bound
#         config = Config(default_config)

#         @config("policy")
#         class PolicyConfig:
#             policy: Type[Policy] = self.de_policy
#             agent: DENeuralUCB = self

#         config("de")(self.de_config)

#         de = DE(config)
#         de.train()

#         return de.population[de.current_best]

#     def train(self, n_trials, n_grad_steps):
#         for t in range(n_trials):
#             a_t = self.de_best_vector()


# class Network(nn.Module):
#     def __init__(self, dim, hidden_size=100):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, hidden_size)
#         self.activate = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         return self.fc3(self.activate(self.fc2(self.activate(self.fc1(x))))


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
        # self.len += 1
        # optimizer = optim.SGD(
        #     self.net.parameters(), lr=lr, weight_decay=self.lamdba / self.len
        # )
        optimizer = optim.SGD(self.net.parameters(), lr=lr)
        for _ in range(n_optim_steps):
            self.net.zero_grad()
            optimizer.zero_grad()
            pred = self.net(self.vec_history)
            loss = self.loss_func(pred, self.reward_history)
            loss.backward()
            optimizer.step()

        # logging.info(loss)
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
        # torch.set_grad_enabled(True)
        mu, g_list = self.compute_activation_and_grad(vec)

        # Lenient version. If mu is higher than the threshold, the risk of incurring regret is small
        # if mu.item() >= self.reward_sample_thresholds[1]:
        #     return mu.item(), None, mu.item(), None

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
