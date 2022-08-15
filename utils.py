import argparse
import json
import os
import logging
import random
from copy import deepcopy
from math import ceil, floor, isnan
from typing import Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from detorch import DE, Policy
from detorch.config import Config, default_config
from scipy.stats.contingency import relative_risk

using_cpu = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device("cuda"):
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    using_cpu = False

num_cpus = len(os.sched_getaffinity(0))

logging.basicConfig(level=logging.INFO)


class PullPolicy(Policy):
    """
    Pull policy for DE. Utility class to do DE
    optimization over dataset feature vectors
    """

    bounds = [0, 1]

    def __init__(self, eval_fn, p, set_existing_vecs, thresh, n_sigmas):
        """
        Args:
            eval_fn (function): _description_
            p (torch.Tensor): probability vector for initialization. Must be equal in length to `set_existing_vecs`
            set_existing_vecs (torch.Tensor): Set of existing vectors to initialize from
            thresh (float): Threshold for ci intersection.
        """
        super().__init__()
        idx = p.multinomial(num_samples=1).item()
        self.params = nn.Parameter(set_existing_vecs[idx].clone(), requires_grad=False)
        self.eval_fn = eval_fn
        self.thresh = thresh
        self.n_sigmas = n_sigmas
        self.mu = None
        self.cb = None
        self.sample_r = None
        self.activation_grad = None
        self.lower_ci_under_thresh = None

    def evaluate(self):
        """Evaluate current `self` (a tensor)
        to find its value according to `eval_fn`"""
        self.transform()
        sample_r, activation_grad, mu, cb = self.eval_fn(self.params.data[None])
        self.activation_grad = activation_grad
        self.sample_r = sample_r.detach().item()
        self.mu = mu
        self.cb = cb
        self.lower_ci_under_thresh = (mu - self.n_sigmas * cb) <= self.thresh
        return self.sample_r

    def transform(self):
        vec = torch.clip(self.params, *PullPolicy.bounds).to(device)
        self.params = nn.Parameter(vec, requires_grad=False)


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
        self.best_model.zero_grad(set_to_none=True)

    @property
    def early_stop(self):
        if self.count >= self.patience:
            return True


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        errors = target - preds

        for i, q in enumerate(self.quantiles):
            losses.append(
                torch.max((q - 1) * errors[:, i], q * errors[:, i]).unsqueeze(1)
            )
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

        return loss


def compute_relative_risk(combi, pop_combis, pop_outcomes):
    # Determined by polypharmacy definition
    if vec.sum() < 5:
        return 0

    vec_indices = (vec.squeeze(0) == 1.0)[0]

    # Get boolean array for exposed and controls
    rows_exposed = torch.where((X[:, vec_indices] == 1).all(dim=1), True, False)
    rows_control = torch.logical_not(rows_exposed)

    n_exposed = rows_exposed.sum()
    n_control = rows_control.sum()
    n_exposed_case = y[rows_exposed].sum()
    n_control_case = y[rows_control].sum()

    rr = (n_exposed_cases / n_exposed) / (n_control_cases / n_control)

    if isnan(rr):
        # Interpreted as 0 by experts
        return 0

    # Clip in a realistic range the RR so we don't end up with infinite RR
    return torch.clip(rr, 0, 10)


def change_to_closest_existing_vector(vec, set_existing_vecs):
    """1-NN search of `vec` in `set_existing_vecs`

    Args:
        vec (torch.Tensor): base vector
        set_existing_vecs (torch.Tensor): neighboring vectors in which to do the search

    Returns:
        torch.Tensor: nearest neighbor of `vec` in `set_existing_vecs`
    """
    dists = torch.norm(vec - set_existing_vecs, dim=1, p=1)
    knn_idx = dists.topk(1, largest=False).indices[0]
    return set_existing_vecs[knn_idx][None, :], knn_idx


def gen_warmup_vecs_and_rewards(n_warmup, combis, risks, p):
    vecs = []
    rewards = []
    for i in range(n_warmup):
        idx = p.multinomial(num_samples=1).item()
        vec = combis[idx]
        reward = risks[idx]
        vecs.append(vec.tolist())
        rewards.append([reward])

    vecs = torch.tensor(vecs)
    rewards = torch.tensor(rewards)
    return vecs, rewards


def find_best_member(
    agent_eval_fn,
    de_config,
    proba,
    set_init_vecs,
    seed,
    ci_thresh,
    threshold,
    n_sigmas_conf,
):
    """
    Run DE to find the best vector for the current agent

    Args:
        agent_eval_fn (function): agent's evaluation function
        de_config (DEConfig): DE Configuration
        proba (torch.Tensor): initialization probas for vectors in DE
        set_init_vecs (torch.Tensor): available vectors to initialize from
        seed (int): seed to set up DE
        ci_thresh (bool): if True, forces reward sorting to get the best member which has a lower ci intersecting with threshold.
        thresh (float): threshold for CI intersection. If ci_thresh = True, then this must be set to a real number.
        n_sigmas_conf (float): Number of sigmas to consider for confidence (sigma-rule) around network activation (mu).Used for stopping exploitation of a known good arm.
    Returns:
        torch.Tensor: Best member from DE's population
    """

    # Reseed DE optim to diversify populations across timesteps
    de_config.seed = seed
    config = Config(default_config)

    @config("policy")
    class PolicyConfig:
        policy: Type[Policy] = PullPolicy
        eval_fn: object = agent_eval_fn
        p: torch.Tensor = proba
        set_existing_vecs: torch.Tensor = set_init_vecs
        thresh: float = threshold
        n_sigmas: float = n_sigmas_conf

    config("de")(de_config)

    de = DE(config)
    de.train()

    # If this is true, then returned best member must
    # have its lower CI bound be lower or equal to the
    # threshold
    best = de.population[de.current_best]

    if ci_thresh and not best.lower_ci_under_thresh:
        sorted_rewards_idx = np.flip(np.argsort(de.rewards))
        sorted_pop = de.population[sorted_rewards_idx]
        sorted_pop_inter = [m.lower_ci_under_thresh for m in sorted_pop]
        first_occ_idx = np.where(sorted_pop_inter)[0][0]
        return sorted_pop[first_occ_idx]
    else:
        return best


def make_deterministic(seed=42):
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Numpy
    np.random.seed(seed)

    # Built-in Python
    random.seed(seed)


def compute_jaccard(found_solution: set, true_solution: set):
    n_in_inter = 0

    intersection = found_solution & true_solution

    n_in_inter = len(intersection)

    return (
        n_in_inter / (len(found_solution) + len(true_solution) - n_in_inter),
        n_in_inter,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a NeuralTS/UCB agent on a given dataset"
    )
    parser.add_argument(
        "-T", "--trials", type=int, required=True, help="Number of trials for the agent"
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="Name of dataset (located in datasets/*)"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        required=True,
        help="Good and bad action threshold",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Set random seed base for training, only affects network initialization",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=128,
        help="Width of the NN (number of neurons)",
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=int,
        default=1,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "-r",
        "--reg",
        type=float,
        default=1,
        help="Regularization factor for the bandit AND weight decay (lambda)",
    )
    parser.add_argument(
        "-e",
        "--exploration",
        type=float,
        default=1,
        help="Exploration multiplier",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Number of epochs / gradient steps (if full GD) in NeuralTS/NeuralUCB",
    )
    parser.add_argument(
        "--lr",
        default=0.01,
        help="Learning rate for SGD / Adam optimizer",
    )

    parser.add_argument(
        "--style",
        type=str,
        default="ts",
        choices=["ts", "ucb"],
        help="Choose between NeuralTS and NeuralUCB to train",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "adam"],
        help="Select SGD or Adam as optimizer for NN",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="saves/ouput/",
        help="Output directory for metrics and agents",
    )
    parser.add_argument(
        "--pop_n_members",
        type=int,
        default=256,
        help="Number of members for the population optimizer",
    )
    parser.add_argument(
        "--pop_n_steps",
        type=int,
        default=16,
        help="Number of step for the population optimizer",
    )
    parser.add_argument(
        "--pop_lr",
        type=float,
        default=1e-2,
        help="Learning rate for the population optimizer (if gradient based)",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for learning (specify -1 for full batch)",
    )
    parser.add_argument(
        "--n_sigmas",
        type=float,
        default=3,
        help="Number of sigmas to consider (sigma-rule) for confidence around a network's given activation",
    )
    parser.add_argument(
        "--ci_thresh",
        action="store_true",
        help="Tells the agent to play the best arm which has an interesecting lower CI with the threshold",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Patience for early stopping during training",
    )
    parser.add_argument(
        "--valtype",
        type=str,
        default="noval",
        help="Strategy for validation set selection",
    )
    parser.add_argument(
        "--nobatchnorm",
        action="store_true",
        help="Use batch norm in neural network",
    )
    parser.add_argument(
        "--lds",
        default="True",
        choices=["True", "sqrt_inv", "False"],
        help="Strategy for label distribution smoothing",
    )
    parser.add_argument(
        "--usedecay",
        action="store_true",
        help="Use weight decay during training",
    )

    parser.add_argument(
        "--ntrain",
        type=int,
        default=-1,
        help="Number of samples to take during training. Can help if there are enough observations to notice a slow down of the training.",
    )

    parser.add_argument(
        "--train_every",
        type=int,
        default=1,
        help="Number of timesteps to play before retraining",
    )

    args = parser.parse_args()
    return args


def do_gradient_optim(agent, n_steps, existing_vecs, lr):
    # Generate a random vector to optimize
    sample_idx = random.randint(
        0, len(existing_vecs) - 1
    )  # random.randint includes upper bound...
    input_vec = existing_vecs[sample_idx][None].clone()
    input_vec.requires_grad = True
    optimizer = torch.optim.Adam([input_vec], lr=lr)

    population = input_vec.detach().clone()
    population_values = []
    # Do n_steps gradient steps, optimizing a noisy sample from the distribution of the input_vec
    for i in range(n_steps):
        # Clear gradients for sample
        optimizer.zero_grad(set_to_none=True)
        agent.net.zero_grad(set_to_none=True)

        # Evaluate
        sample_r, g_list, mu, cb = agent.get_sample(input_vec)
        # Clear gradient from sampling because a backprop happens in there
        optimizer.zero_grad(set_to_none=True)
        agent.net.zero_grad(set_to_none=True)

        # Record input_vecs and values in the population
        population_values.append(sample_r.item())

        # Backprop
        sample_r = -sample_r
        sample_r.backward()
        # print(agent.net.fc1.weight.grad)
        # input()
        optimizer.step()

        population = torch.cat((population, input_vec.detach().clone()))

    # Clear gradients for sample
    optimizer.zero_grad(set_to_none=True)
    agent.net.zero_grad(set_to_none=True)
    # Record final optimized input_vecs in population since they're the last optimizer steps product
    sample_r, g_list, mu, cb = agent.get_sample(input_vec)

    population_values.append(sample_r.item())

    # Clean up grad before exiting
    optimizer.zero_grad(set_to_none=True)
    agent.net.zero_grad(set_to_none=True)
    population_values = torch.tensor(population_values)

    # Find the best generated vector
    max_idx = torch.argmax(population_values)
    best_vec = population[max_idx]

    # Coerce to an existing vector via L1 norm
    a_t, idx = change_to_closest_existing_vector(best_vec, existing_vecs)
    # print(a_t.shape)
    _, g_list = agent.compute_activation_and_grad(a_t)
    # print(best_vec)
    # print(a_t)
    # input()
    return a_t, idx, g_list


def load_dataset(dataset_name, path_to_dataset="datasets"):

    dataset = pd.read_csv(f"{path_to_dataset}/combinations/{dataset_name}.csv")

    with open(f"{path_to_dataset}/patterns/{dataset_name}.json", "r") as f:
        patterns = json.load(f)
    # Remove last 3 columns that are risk, inter, dist
    combis = dataset.iloc[:, :-3]

    # Retrieve risks
    risks = dataset.iloc[:, -3]

    n_obs, n_dim = combis.shape

    pat_vecs = torch.tensor(
        [patterns[f"pattern_{i}"]["pattern"] for i in range(len(patterns))]
    )
    combis, risks = (
        torch.tensor(combis.values).float(),
        torch.tensor(risks.values).unsqueeze(1).float(),
    )

    return combis, risks, pat_vecs, n_obs, n_dim


def get_data_splits(combis, risks, valtype="extrema"):
    if valtype == "extrema":
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
    elif valtype == "bins":
        # Use bins for validation set, so we have a wide spread
        bin_size = 0.1
        min_range = 0
        max_range = 4
        bins_edges = [
            round(min_range + (i * bin_size), 1)
            for i in range(int((max_range - min_range) / bin_size) + 1)
        ]
        X_val = []
        y_val = []
        # Put one observation per bin
        for i in range(len(bins_edges) - 1):
            lower_bound = bins_edges[i]
            upper_bound = bins_edges[i + 1]
            bigger_than = risks >= lower_bound
            smaller_than = risks < upper_bound
            inbound = torch.cat((bigger_than, smaller_than), dim=1).all(dim=1)
            idx = torch.where(inbound)[0]
            if len(idx) > 0:
                idx = idx[0]
                X_val.append(combis[idx])
                y_val.append(risks[idx])
                combis = torch.cat((combis[:idx], combis[idx + 1 :]))
                risks = torch.cat((risks[:idx], risks[idx + 1 :]))
        X_val = torch.stack(X_val)
        y_val = torch.stack(y_val)
    if valtype == "noval":
        X_val = None
        y_val = None

    X_train, y_train = combis, risks

    return X_train, y_train, X_val, y_val


def compute_metrics(
    agent,
    combis,
    thresh,
    pat_vecs,
    true_sol_idx,
    n_sigmas,
    all_flagged_combis_idx,
    all_flagged_pats_idx,
):
    """Compute metrics for combination test

    Args:
        agent (OptimNeuralTS): the bandit agent
        combis (torch.Tensor): all possible combinations of Rx in the dataset
        thresh (float): threshold of risk
        pat_vecs (torch.Tensor): pattern vectors used to generate dataset
        true_sol (torch.Tensor): true solution of the dataset
        n_sigmas (float): number of sigmas to consider (sigma-rule sense)
        all_flagged_combis (torch.Tensor): all previously flagged combinations
        all_flagged_pats (torch.Tensor): all previously flagged patterns

    Returns:
        tuple: tuple of metrics and updated tensors in the following order:
        jaccard for current step,
        ratio_app for current step,
        percent_found_pat for current step,
        n_inter for current step,
        jaccard for all steps so far,
        ratio_app for all steps so far,
        percent_found_pat for all steps so far,
        n_inter for all steps so far,
        updated all flagged combis,
        updated all flagged pats,
    """

    # Parmis tous les vecteurs "existant", lesquels je trouve ? (Jaccard, ratio_app)
    sol_idx, _, _ = agent.find_solution_in_vecs(combis, thresh, n_sigmas)

    all_flagged_combis_idx.update(sol_idx)

    # Parmis les patrons dangereux (ground truth), combien j'en trouve tels quels
    sol_pat_idx, _, _ = agent.find_solution_in_vecs(pat_vecs, thresh, n_sigmas)

    all_flagged_pats_idx.update(sol_pat_idx)

    # À quel point ma solution trouvée parmis les vecteurs du dataset est similaire à la vraie solution
    jaccard, n_inter = compute_jaccard(
        sol_idx, true_sol_idx
    )  # Jaccard for the current step

    jaccard_all, n_inter_all = compute_jaccard(
        all_flagged_combis_idx, true_sol_idx
    )  # Jaccard for all steps before + this one if we keep all previous solutions

    # Combien de patrons tels quels j'ai flag ?
    percent_found_pat = len(sol_pat_idx) / len(pat_vecs)  # For this step
    percent_found_pat_all = len(all_flagged_pats_idx) / len(
        pat_vecs
    )  # For all previous steps and this one

    # A quel point ma solution trouvee parmis les vecteurs du dataset est dans la vraie solution
    if len(sol_idx) == 0:
        ratio_app = float("nan")
    else:
        ratio_app = n_inter / len(sol_idx)

    if len(all_flagged_combis_idx) == 0:
        ratio_app_all = float("nan")
    else:
        ratio_app_all = n_inter_all / len(all_flagged_combis_idx)

    return (
        jaccard,
        ratio_app,
        percent_found_pat,
        n_inter,
        jaccard_all,
        ratio_app_all,
        percent_found_pat_all,
        n_inter_all,
        all_flagged_combis_idx,
        all_flagged_pats_idx,
    )


def discretize_targets(targets, factor):
    discrete_risks = torch.floor(targets * factor) / factor
    discrete_risks = np.round(discrete_risks.cpu().numpy(), decimals=1)

    return discrete_risks


def build_histogram(targets, factor, bin_size):
    # Determine the bin edges
    min_bin = floor(min(targets) * factor) / factor
    max_bin = ceil(max(targets) * factor) / factor
    # Handles case where maximum is exactly on the edge of the last bin
    if round(max(targets).item(), 1) == max_bin:
        max_bin += bin_size

    n_bins = round((max_bin - min_bin) / 0.1)  # Deal with poor precision in Python...
    list_bin_edges = np.around(
        [min_bin + (bin_size * i) for i in range(n_bins + 1)], 1
    ).astype("float32")
    bin_edges = torch.from_numpy(list_bin_edges)

    # Build discretized distribution with histogram
    hist = torch.histogram(targets.cpu(), bin_edges)
    return hist, n_bins, list_bin_edges


def gaussian_fn(size, std):
    n = torch.arange(0, size) - (size - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-(n**2) / sig2)
    return w


def get_model_selection_loss(net, X_val, y_val, loss_fn):
    with torch.no_grad():
        pred = net(X_val)
        loss = loss_fn(pred, y_val)
    return loss
