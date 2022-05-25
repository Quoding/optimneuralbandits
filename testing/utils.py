import argparse
import json
import os
import random
from math import isnan
from typing import Type
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from detorch import DE, Policy, Strategy
from detorch.config import Config, default_config
from scipy.stats.contingency import relative_risk

device = torch.device("cuda")
logging.basicConfig(level=logging.INFO)
torch.set_default_tensor_type("torch.cuda.FloatTensor")


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


class PullPolicy(Policy):
    """
    Pull policy for DE. Utility class to do DE
    optimization over dataset feature vectors
    """

    bounds = [0, 1]

    def __init__(self, eval_fn, p, set_existing_vecs):
        super().__init__()
        idx = p.multinomial(num_samples=1).item()
        self.params = nn.Parameter(set_existing_vecs[idx].clone(), requires_grad=False)
        self.eval_fn = eval_fn
        self.sample_r = None
        self.activation_grad = None

    def evaluate(self):
        """Evaluate current `self` (a tensor)
        to find its value according to `eval_fn`"""
        self.transform()
        sample_r, activation_grad, _, _ = self.eval_fn(self.params.data)
        self.activation_grad = activation_grad
        self.sample_r = sample_r
        return sample_r

    def transform(self):
        vec = torch.clip(self.params, *PullPolicy.bounds).to(device)
        self.params = nn.Parameter(vec, requires_grad=False)


class DEConfig:
    n_step: int = 16
    population_size: int = 32
    differential_weight: float = 1
    crossover_probability: float = 0.9
    strategy: Strategy = Strategy.best2bin
    seed: int = "doesn't matter"


def compute_relative_risk(vec, X, y):
    # Determined by polypharmacy definition
    if vec.sum() < 5:
        return 0

    vec_indices = torch.where(vec == 1)[0]

    # Exposed
    rows_exposed = torch.where((X[:, vec_indices] == 1).all(axis=1))[0]
    rows_control = torch.where((X[:, vec_indices] == 0).any(axis=1))[0]
    rows_exposed_case = torch.where(y[rows_exposed] == 1)[0]
    rows_control_case = torch.where(y[rows_control] == 1)[0]

    n_exposed = len(rows_exposed)
    n_exposed_case = len(rows_exposed_case)
    n_control = len(rows_control)
    n_control_case = len(rows_control_case)
    rr = relative_risk(
        n_exposed_case, n_exposed, n_control_case, n_control
    ).relative_risk

    if isnan(rr):
        # Interpreted as 0 by experts
        return 0

    elif rr == float("inf"):
        return 10  # Return something really big, but not inf so it doesn't break the regression

    return rr


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
    return set_existing_vecs[knn_idx], knn_idx


def gen_warmup_vecs_and_rewards(n_warmup, X, y, p):
    vecs = []
    rewards = []
    for i in range(n_warmup):
        idx = p.multinomial(num_samples=1).item()
        vec = X[idx]
        reward = compute_relative_risk(vec, X, y)
        vecs.append(vec.tolist())
        rewards.append([reward])

    vecs = torch.tensor(vecs)
    rewards = torch.tensor(rewards)
    return vecs, rewards


def find_best_member(agent_eval_fn, de_config, proba, set_init_vecs, seed):
    """Run DE to find the best vector for the current agent

    Args:
        agent_eval_fn (function): agent's evaluation function
        de_config (DEConfig): DE Configuration
        proba (torch.Tensor): initialization probas for vectors in DE
        set_init_vecs (torch.Tensor): available vectors to initialize from
        seed (int): seed to set up DE

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

    config("de")(de_config)

    de = DE(config)
    de.train()

    return de.population[de.current_best]


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


def compute_jaccard(found_solution, true_solution):
    found_sol_list = found_solution.tolist()
    true_sol_list = true_solution.tolist()

    n_in_inter = 0

    for vec in found_sol_list:
        n_in_inter += vec in true_sol_list

    return (
        n_in_inter / (len(found_solution) + len(true_solution) - n_in_inter),
        n_in_inter,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train an agent on a given dataset")
    parser.add_argument(
        "-T", "--trials", type=int, required=True, help="Number of trials for the agent"
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="Name of dataset (located datasets/*"
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
        help="Set random seed base for training (will be added to current trial number to diversify DE)",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=100,
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
        help="Regularization factor",
    )
    parser.add_argument(
        "-e",
        "--exploration",
        type=float,
        default=1,
        help="Exploration multiplier",
    )
    parser.add_argument(
        "--n_optim_steps",
        type=int,
        default=100,
        help="Regularization factor for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate for SGD / Adam optimizer",
    )

    parser.add_argument(
        "--style",
        type=str,
        default="ts",
        choices=["ts", "ucb"],
        help="Learning rate for SGD / Adam optimizer",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
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
    args = parser.parse_args()
    return args


def load_dataset(dataset_path):
    dataset = pd.read_csv("datasets/combinations/" + dataset_path + ".csv")

    with open("datasets/patterns/" + dataset_path + ".json", "r") as f:
        patterns = json.load(f)

    features = dataset.iloc[:, :-3]
    risks = dataset.iloc[:, -3]

    n_obs, n_dim = features.shape

    return features, risks, patterns, n_obs, n_dim


def compute_metrics(agent, combis, thresh, pat_vecs, true_sol):
    # Parmis tous les vecteurs existant, lesquels je trouve ? (Jaccard, ratio_app)
    sol = agent.find_solution_in_vecs(combis, thresh)
    # Parmis les patrons insérés, combien j'en trouve tels quels
    sol_pat = agent.find_solution_in_vecs(pat_vecs, thresh)
    # À quel point ma solution trouvée parmis les vecteurs du dataset est similaire à la vraie solution
    jaccard, n_inter = compute_jaccard(sol, true_sol)
    # Combien de patrons tels quels j'ai flag ?
    percent_found_pat = len(sol_pat) / len(pat_vecs)
    # A quel point ma solution trouvee parmis les vecteurs du dataset est dans la vraie solution
    if len(sol) == 0:
        ratio_app = 0
    else:
        ratio_app = n_inter / len(sol)

    return jaccard, ratio_app, percent_found_pat, n_inter
