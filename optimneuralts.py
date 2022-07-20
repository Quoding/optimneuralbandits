import logging
import types
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets import ReplayDataset, ValidationReplayDataset
from utils import EarlyStopping, get_validation_loss

logging.basicConfig(level=logging.INFO)


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
        patience=25,
        use_lds=True,
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
        shuffle = True
        sampler = None
        if use_lds:
            w = self.train_dataset.get_weights(reweight="sqrt_inv")
            sampler = WeightedRandomSampler(w, num_samples=len(self.train_dataset))
            shuffle = False

        loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            sampler=sampler,
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
            n_sigmas (int): Number of sigmas to consider (sigma rule)
        Returns:
            tensor: torch.Tensor of torch.Tensor of the solution
        """

        solution = []
        mus = []
        sigmas = []
        self.net.eval()
        for vec in vecs:
            mu, g_list = self.compute_activation_and_grad(vec[None])
            sigma = torch.sum(g_list * g_list / self.U)
            sigma = torch.sqrt(self.lambda_ * sigma)

            if (mu - n_sigmas * sigma).item() > thresh:
                solution.append(vec)

            mus.append(mu.item())
            sigmas.append(n_sigmas * sigma.item())
        if solution:
            solution = torch.stack(solution)
        else:
            solution = torch.tensor([])
        self.net.train()

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
