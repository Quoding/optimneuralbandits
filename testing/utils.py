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
from torchviz import make_dot
from detorch import DE, Policy, Strategy
from detorch.config import Config, default_config
from scipy.stats.contingency import relative_risk

device = torch.device("cuda")
logging.basicConfig(level=logging.INFO)
torch.set_default_tensor_type("torch.cuda.FloatTensor")


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
        sample_r, activation_grad, mu, cb = self.eval_fn(self.params.data)
        self.activation_grad = activation_grad
        self.sample_r = sample_r.detach().item()
        self.mu = mu
        self.cb = cb
        self.lower_ci_under_thresh = (mu - 3 * cb) <= self.thresh
        return self.sample_r

    def transform(self):
        vec = torch.clip(self.params, *PullPolicy.bounds).to(device)
        self.params = nn.Parameter(vec, requires_grad=False)


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
        n_sigmas_conf (float): Number of sigmas to consider for confidence (sigma-rule) around network activation (mu)
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
        default=2,
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
        default=10,
        help="Number of epochs / gradient steps (if full SGD) in NeuralTS/NeuralUCB",
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
        help="Choose between NeuralTS and NeuralUCB to train",
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
    parser.add_argument(
        "--pop_n_members",
        type=int,
        default=32,
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
    args = parser.parse_args()
    return args


def do_gradient_optim(agent_eval_fn, n_steps, existing_vecs, lr):
    # Generate a random vector to optimize
    input_vec = torch.randint(0, 2, size=(1, existing_vecs.shape[1])).float()
    input_vec.requires_grad = True
    optimizer = torch.optim.Adam([input_vec], lr=lr)

    population = input_vec.detach().clone()
    population_values = []

    # Do n_steps gradient steps, optimizing a noisy sample from the distribution of the input_vec
    for i in range(n_steps):
        optimizer.zero_grad()
        # Evaluate
        sample_r, g_list, mu, cb = agent_eval_fn(input_vec)
        print(sample_r)
        dot = make_dot(sample_r)
        dot.format = "svg"
        dot.render()
        # Record input_vecs and values in the population
        population_values.append(sample_r.item())

        # Backprop
        sample_r = -sample_r
        sample_r.backward()
        optimizer.step()

        population = torch.cat((population, input_vec.detach().clone()))

    # Record final optimized input_vecs in population since they're the last optimizer steps product
    sample_r, g_list, mu, cb = agent_eval_fn(input_vec)

    population_values.append(sample_r.item())

    # Clean up grad before exiting
    optimizer.zero_grad()

    population_values = torch.tensor(population_values)

    # Find the best generated vector
    max_idx = torch.argmax(population_values)
    best_vec = population[max_idx]

    # Coerce to an existing vector via L1 norm
    a_t, idx = change_to_closest_existing_vector(best_vec, existing_vecs)
    sample_r, g_list, mu, cb = agent_eval_fn(a_t)
    input()
    return a_t, idx, g_list


def load_dataset(dataset_name, path_to_dataset=None):
    if path_to_dataset is None:
        dataset = pd.read_csv(f"datasets/combinations/{dataset_name}.csv")

        with open(f"datasets/patterns/{dataset_name}.json", "r") as f:
            patterns = json.load(f)

    else:
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


def compute_metrics(agent, combis, thresh, pat_vecs, true_sol, n_sigmas):
    # Parmis tous les vecteurs existant, lesquels je trouve ? (Jaccard, ratio_app)
    sol, _, _ = agent.find_solution_in_vecs(combis, thresh, n_sigmas)
    # Parmis les patrons dangereux (ground truth), combien j'en trouve tels quels
    sol_pat, _, _ = agent.find_solution_in_vecs(pat_vecs, thresh, n_sigmas)
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


class Network(nn.Module):
    """Deprecated, maintained here for compatibility"""

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
