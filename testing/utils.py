import torch
import torch.nn as nn
from detorch import DE, Policy, Strategy
from detorch.config import default_config, Config
from typing import Type
from scipy.stats.contingency import relative_risk
import numpy as np
from math import isnan
import random
import os

device = torch.device("cuda")


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
    bounds = [0, 1]

    def __init__(self, eval_fn, p, set_existing_vecs):
        super().__init__()
        idx = p.multinomial(num_samples=1).item()
        self.params = nn.Parameter(set_existing_vecs[idx].clone(), requires_grad=False)
        self.eval_fn = eval_fn
        self.sample_r = None
        self.activation_grad = None

    def evaluate(self):
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
    seed: int = 42


def risk_reward_fn(vec, X, y):
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
    dists = torch.norm(vec - set_existing_vecs, dim=1, p=1)
    knn_idx = dists.topk(1, largest=False).indices[0]
    return set_existing_vecs[knn_idx]


def gen_warmup_vecs_and_rewards(n_warmup, X, y, p, set_existing_vecs):
    vecs = []
    rewards = []
    for i in range(n_warmup):
        idx = p.multinomial(num_samples=1).item()
        vec = set_existing_vecs[idx]
        reward = risk_reward_fn(vec, X, y)
        vecs.append(vec.tolist())
        rewards.append([reward])

    vecs = torch.tensor(vecs)
    rewards = torch.tensor(rewards)
    return vecs, rewards


def reseed_de(de_config):
    # Bypass default config of DE and modify seed so populations are different from invocation to invocation
    random.seed(os.urandom(100))
    de_config.seed = random.randint(0, 100000)


def find_best_member(agent_eval_fn, de_config, proba, set_init_vecs):
    reseed_de(de_config)
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
