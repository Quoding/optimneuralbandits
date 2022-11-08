import logging
import types
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets import ReplayDataset, ValidationReplayDataset, FastTensorDataLoader
from utils import EarlyStopping, get_model_selection_loss

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
        valtype="noval",
    ):
        self.net = net
        self.lambda_ = lambda_
        self.total_param = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        self.len = 0
        self.U = lambda_ * torch.ones((self.total_param,))
        self.nu = nu
        self.style = style
        self.sampletype = sampletype

        self.loss_func = nn.MSELoss()
        self.train_dataset = ReplayDataset()
        self.val_dataset = ValidationReplayDataset(valtype=valtype)
        self.valtype = valtype

        optimizers = {"sgd": optim.SGD, "adam": optim.Adam}
        # Keep optimizer separate from DENeuralTS class to tune lr as we go through timesteps if we so desire
        self.optimizer_class = optimizers[optim_string]

    def compute_activation_and_grad(self, vec):
        self.net.zero_grad(set_to_none=True)
        mu = self.net(vec)
        mu.backward(retain_graph=True)
        g_list = torch.cat([p.grad.flatten() for p in self.net.parameters()])
        return mu, g_list

    def get_sample(self, vec):
        mu, g_list = self.compute_activation_and_grad(vec)
        sigma = torch.tensor(0.0)

        sigma = sigma + torch.sum(g_list * g_list / self.U)
        sigma = self.nu * torch.sqrt(self.lambda_ * sigma)

        if self.sampletype == "f":
            # Exploration is generated by the dropout value, reflected in mu
            # mu = mu
            return mu, g_list, mu.item(), sigma.detach().item()

        if self.style == "ts":
            sample_r = torch.distributions.Normal(mu.view(-1), sigma.view(-1)).rsample()
        elif self.style == "ucb":
            sample_r = mu.view(-1) + sigma.view(-1)

        return sample_r, g_list, mu.detach().item(), sigma.detach().item()

    def train(
        self,
        n_epochs,
        lr=1e-2,
        batch_size=-1,
        patience=25,
        lds=True,
        n_train=-1,
        use_decay=False,
    ):
        n_dataset = len(self.train_dataset)
        # For full batch grad descent
        if batch_size == -1:
            batch_size = n_dataset
        if n_train == -1:
            n_train = n_dataset
        # Setup
        # self.len += 1
        self.len = n_dataset
        weight_decay = use_decay * (self.lambda_ / self.len)

        if lr == "plateau":
            optimizer = torch.optim.Adam(
                self.net.parameters(), lr=0.01, weight_decay=weight_decay
            )
            sched = ReduceLROnPlateau(optimizer, "min", patience=patience // 2)
        else:
            optimizer = self.optimizer_class(
                self.net.parameters(), lr=float(lr), weight_decay=weight_decay
            )

        if lds:
            self.train_dataset.update_weights(reweight=lds)

        self.train_dataset.update_dataset()

        loader = FastTensorDataLoader(
            self.train_dataset.training_features,
            self.train_dataset.training_rewards,
            batch_size=batch_size,
            shuffle=True,
        )

        early_stop = EarlyStopping(patience)

        # Choose whether we do model selection based on training set loss or validation set loss
        if self.valtype == "noval":
            X_val, y_val = (
                self.train_dataset.features,
                self.train_dataset.rewards,
            )
        else:
            X_val, y_val = self.val_dataset.features, self.val_dataset.rewards
        # Train loop
        for _ in range(n_epochs):
            for X, y in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = self.net(X)
                loss = self.loss_func(pred, y)
                loss.backward()
                optimizer.step()

            self.net.eval()
            stored_loss = get_model_selection_loss(
                self.net, X_val, y_val, self.loss_func
            )
            self.net.train()
            early_stop(stored_loss, self.net)

            if lr == "plateau":
                sched.step(stored_loss)

            if early_stop.early_stop:
                break

        optimizer.zero_grad(set_to_none=True)

        # Set the net to the best in model selection loss we've seen
        # Deep copy to ensure no remaining references to the early_stop object
        self.net = deepcopy(early_stop.best_model)
        return stored_loss.detach().item()

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

        solution_idx = set()
        mus = []
        sigmas = []
        # First pass, weed out vectors with small activs so we don't waste time in the loop to extract solutions
        activs = self.net(vecs)
        possible_solution_vecs_idx = torch.where(activs > thresh)[0].tolist()
        possible_solution_vecs = vecs[possible_solution_vecs_idx]

        for i, vec in zip(possible_solution_vecs_idx, possible_solution_vecs):
            mu, g_list = self.compute_activation_and_grad(vec[None])
            sigma = torch.sum(g_list * g_list / self.U)
            sigma = torch.sqrt(self.lambda_ * sigma)

            if (mu - n_sigmas * sigma).item() > thresh:
                solution_idx.add(i)

            mus.append(mu.item())
            sigmas.append(n_sigmas * sigma.item())

        return (solution_idx, mus, sigmas)


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
            sample_r = torch.tensor([0.0])
            torch.nn.init.trunc_normal_(
                sample_r, mu.view(-1), sigma.view(-1), *self.reward_sample_thresholds
            )
        # torch.set_grad_enabled(False)
        return sample_r, g_list, mu.detach().item(), cb.item()
