# %%
import os
import random
import sys
from math import sqrt
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from detorch import DE, Policy, Strategy
from detorch.config import Config, default_config
from torch.nn.functional import normalize

sys.path.append("..")
sys.path.append("viz")
import logging

import viz_config

from networks import NetworkDropout
from optimneuralts import LenientOptimNeuralTS, OptimNeuralTS
from utils import do_gradient_optim

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config(default_config)
bounds = [-2.5, 1.5]
theta = torch.Tensor([0, 3, -2, -4, 1, 1]).to(device)
d = 1
torch.set_default_tensor_type("torch.cuda.FloatTensor")

# %% [markdown]
# # Classes

# %%
class PullPolicy(Policy):
    def __init__(self, eval_fn):
        super().__init__()
        self.point = torch.FloatTensor(1).uniform_(*bounds).to(device)
        self.params = nn.Parameter(self.point, requires_grad=False)
        self.eval_fn = eval_fn
        self.ucb = None

    def evaluate(self):
        self.transform()
        ucb, activation_grad, _, _ = self.eval_fn(self.point)
        ucb = ucb.detach().item()
        self.activation_grad = activation_grad
        self.ucb = ucb
        # logging.info(self.point)
        # logging.info(ucb)
        return ucb

    def transform(self):
        self.point = torch.clip(self.params, *bounds).to(device)
        self.params = nn.Parameter(self.point, requires_grad=False)
        # return generate_feature_vector_from_point(self.point)


class DEConfig:
    n_step: int = 3
    population_size: int = 60
    differential_weight: float = 0.8
    crossover_probability: float = 0.9
    strategy: Strategy = Strategy.best1bin
    seed: int = "does not matter"


# %% [markdown]
# # Utility

# %%
def compute_jaccard(found_solution: set, true_solution: set):
    n_in_inter = 0

    intersection = found_solution & true_solution

    n_in_inter = len(intersection)

    return (
        n_in_inter / (len(found_solution) + len(true_solution) - n_in_inter),
        n_in_inter,
    )


def make_deterministic(seed):
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Numpy
    np.random.seed(seed)

    # Built-in Python
    random.seed(seed)


def project_point(point):
    return torch.tensor([1, point, point**2, point**3, point**4, point**5]).to(
        device
    )


def reward_fn(point, add_noise=True):
    noise = torch.normal(0, 1, (1,)).item()

    vec = project_point(point)
    value = theta @ vec.T

    return value + add_noise * noise


def gen_warmup_vecs_and_rewards(n_warmup):
    vecs = torch.tensor([])
    rewards = torch.tensor([])
    for i in range(n_warmup):
        point = torch.FloatTensor(1).uniform_(bounds[0], bounds[1]).to(device)
        reward = torch.tensor([reward_fn(point)])

        vecs = torch.cat((vecs, point))
        rewards = torch.cat((rewards, reward))
    vecs = vecs.view((n_warmup, -1))
    rewards = rewards.view((n_warmup, -1))
    return vecs, rewards


def plot_estimate(agent, trial, fn=None, title=""):
    n_points = 1000
    x = np.linspace(-2.5, 1.5, n_points)
    x_vec = []
    y = []
    for x_point in x:
        y.append(reward_fn(float(x_point), add_noise=False).cpu().numpy())

    y_pred = []
    cbs = []
    ucbs = []
    x_tns = torch.from_numpy(x)
    x_tns = x_tns.view(n_points, 1).float()

    for point in x_tns:
        sample, _, activ, cb = agent.get_sample(point.to(device))
        y_pred.append(activ)
        cbs.append(3 * cb)
        ucbs.append(sample)

    y_pred = np.array(y_pred)
    cbs = np.array(cbs)

    point_played = agent.train_dataset.features.squeeze(0).cpu().numpy()
    rewards_rec = agent.train_dataset.rewards.squeeze(0).cpu().numpy()
    n_played = point_played.shape[0]

    # plt.ylim(-20, 10)
    # plt.ylim(-25, 10)
    plt.plot(x, y, color="tab:blue", label="Vraie fonction")
    plt.plot(x, y_pred, color="tab:orange", label="Estimation de la fonction")
    # plt.fill_between(x, y_pred, ucbs, color='tab:red', alpha=0.3)
    plt.fill_between(
        x,
        y_pred + cbs,
        y_pred - cbs,
        alpha=0.3,
        color="tab:orange",
        zorder=-1,
        label="Intervalle de confiance",
    )
    plt.plot(
        x,
        [0] * n_points,
        color="black",
        linestyle="dashed",
        label="Seuil bonne/mauvaise action",
    )
    plt.scatter(
        point_played[:n_played],
        rewards_rec[:n_played],
        color="black",
        alpha=0.5,
        label="Points joués précédemment",
    )
    plt.scatter(
        point_played[-1], rewards_rec[-1], color="green", label="Dernier point joué"
    )

    plt.title(title)

    plt.xlabel("$x$")
    plt.ylabel("$y$")

    plt.legend()
    if fn is None:
        filename = f"viz/images/exp_poly/regTS_{n_trials}_trials_expl_{exploration_mult}_trial_{trial}.png"
    else:
        filename = f"viz/images/exp_poly/{fn}.png"
    plt.savefig(filename)

    plt.clf()


def find_best_member(eval_fn, de_config, seed):
    de_config.seed = seed
    config = Config(default_config)

    @config("policy")
    class PolicyConfig:
        policy: Type[Policy] = PullPolicy
        eval_fn: object = agent.get_sample

    config("de")(de_config)

    de = DE(config)
    de.train()

    return de.population[de.current_best]


# %% [markdown]
# # Train all runs, for all algos and every exploration

# %%
x_arr = np.linspace(-2.5, 1.5, 1000)
x_arr = x_arr.reshape(1000, 1)
x = torch.from_numpy(x_arr).to(device).float()
y = []
for point in x:
    y.append(reward_fn(point, add_noise=False).cpu().numpy())

y = np.array(y)
true_sol_idx = np.where(y >= 0)[0]
true_sol = x[true_sol_idx].to(device).float()
true_sol_idx = set(true_sol_idx.tolist())

n_true_sol = len(true_sol)
logging.info(n_true_sol)
n_sigmas = 3

# %%
metrics_dict = {}
algos = ["UCB", "regTS"]
exploration_mults = [1, 10]
logging.info(metrics_dict)
max_n_steps = 10
for algo in algos:
    metrics_dict[algo] = {}
    for exploration_mult in exploration_mults:
        metrics_dict[algo][str(exploration_mult)] = {}
        metrics_dict[algo][str(exploration_mult)]["jaccards"] = []
        metrics_dict[algo][str(exploration_mult)]["percent_inter"] = []
        metrics_dict[algo][str(exploration_mult)]["percent_found"] = []
        metrics_dict[algo][str(exploration_mult)]["fails"] = 0

        b = 0

        for i in range(b, 50):
            logging.info(f"at algo: {algo}, expl_mult: {exploration_mult}, run: {i}")
            make_deterministic(seed=i)
            vecs, rewards = gen_warmup_vecs_and_rewards(10)
            vecs, rewards = vecs.to(device), rewards.to(device)
            n_trials = 500
            width = 100
            net = NetworkDropout(d, width).to(device)
            reg = 1
            delay = 0
            reward_fn = reward_fn
            de_config = DEConfig
            de_policy = PullPolicy
            lr = 1e-2

            if algo == "UCB":
                agent = OptimNeuralTS(
                    net, nu=exploration_mult, lambda_=reg, style="ucb"
                )
            elif algo == "regTS":
                agent = OptimNeuralTS(net, nu=exploration_mult, lambda_=reg, style="ts")
            elif algo == "lenientTS":
                agent = LenientOptimNeuralTS(
                    [float("-inf"), 0],
                    net=net,
                    nu=exploration_mult,
                    lambda_=reg,
                )

            vecs, rewards = gen_warmup_vecs_and_rewards(10)
            vecs, rewards = vecs.to(device), rewards.to(device)
            # Warmup
            for vec in vecs:
                activ, grad = agent.compute_activation_and_grad(vec)
                agent.U += grad * grad

            agent.train_dataset.set_(vecs, rewards)
            agent.net.train()
            agent.train(max_n_steps, patience=max_n_steps, lds=False)
            agent.net.eval()

            # Playing
            for j in range(n_trials):
                a_t, idx, best_member_grad = do_gradient_optim(
                    agent, 3 * 60, x, lr=1e-2, bounds=bounds
                )
                r_t = reward_fn(a_t).unsqueeze(0).unsqueeze(0)

                agent.U += best_member_grad * best_member_grad

                agent.train_dataset.add(a_t, r_t)

                agent.net.train()
                agent.train(max_n_steps, patience=max_n_steps, lds=False)
                agent.net.eval()

            sol, _, _ = agent.find_solution_in_vecs(x, 0, n_sigmas=n_sigmas)
            # sol = sol.to(device)
            n_sol = len(sol)

            jaccard, n_inter = compute_jaccard(sol, true_sol_idx)
            percent_found = n_inter / n_true_sol
            if n_sol == 0:
                percent_inter = 0
            else:
                percent_inter = n_inter / n_sol

            metrics_dict[algo][str(exploration_mult)]["jaccards"].append(jaccard)
            metrics_dict[algo][str(exploration_mult)]["percent_inter"].append(
                percent_inter
            )
            metrics_dict[algo][str(exploration_mult)]["percent_found"].append(
                percent_found
            )

            if n_sol == 0:
                logging.info(f"Found no solution for run {i}")
                metrics_dict[algo][str(exploration_mult)]["fails"] += 1

                plot_estimate(
                    agent,
                    n_trials,
                    fn=f"grad_no_sol_{algo}_expl_{exploration_mult}_100_trials_seed_{i}",
                )

            logging.info(
                f"jaccard: {jaccard}, percent_inter: {percent_inter}, percent_found: {percent_found}"
            )

torch.save(metrics_dict, f"metrics_dict_exp_synth_{max_n_steps}_basic.pth")


# %%
logging.info("GRAD_REGULAR")
logging.info("GRAD_REGULAR")
logging.info("GRAD_REGULAR")
for algo in metrics_dict.keys():
    logging.info(f"Algo: {algo}")
    for expl_mult in metrics_dict[algo].keys():
        logging.info(f"Expl mult: {expl_mult}")
        for metric in metrics_dict[algo][expl_mult].keys():
            logging.info(f"Metric: {metric}")
            logging.info(
                f"mean: {np.mean(metrics_dict[algo][expl_mult][metric])} +- {np.std(metrics_dict[algo][expl_mult][metric])} "
            )
            logging.info(
                f"interval:  [{np.min(metrics_dict[algo][expl_mult][metric])}, {np.max(metrics_dict[algo][expl_mult][metric])}]"
            )
            logging.info("============================================")
        # if 0 in metrics_dict[algo][expl_mult]:
        #     logging.info(f'rerun {metrics_dict[algo][expl_mult][metric].index(0)} with plotting')

    logging.info("============================================")


# %% [markdown]
# ## Test avec weight decay

# %%
metrics_dict = {}
# algos = ["UCB", "regTS", "lenientTS"]
algos = ["UCB", "regTS"]
exploration_mults = [1, 10]
logging.info(metrics_dict)
max_n_steps = 10
for algo in algos:
    metrics_dict[algo] = {}
    for exploration_mult in exploration_mults:
        metrics_dict[algo][str(exploration_mult)] = {}
        metrics_dict[algo][str(exploration_mult)]["jaccards"] = []
        metrics_dict[algo][str(exploration_mult)]["percent_inter"] = []
        metrics_dict[algo][str(exploration_mult)]["percent_found"] = []
        metrics_dict[algo][str(exploration_mult)]["fails"] = 0

        b = 0

        for i in range(b, 50):
            logging.info(f"at algo: {algo}, expl_mult: {exploration_mult}, run: {i}")
            make_deterministic(seed=i)
            vecs, rewards = gen_warmup_vecs_and_rewards(10)
            vecs, rewards = vecs.to(device), rewards.to(device)
            n_trials = 500
            width = 100
            net = NetworkDropout(d, width).to(device)
            reg = 1
            reward_fn = reward_fn
            de_config = DEConfig
            de_policy = PullPolicy
            lr = 1e-2

            if algo == "UCB":
                agent = OptimNeuralTS(
                    net, nu=exploration_mult, lambda_=reg, style="ucb"
                )
            elif algo == "regTS":
                agent = OptimNeuralTS(net, nu=exploration_mult, lambda_=reg, style="ts")
            elif algo == "lenientTS":
                agent = LenientOptimNeuralTS(
                    [float("-inf"), 0],
                    net=net,
                    nu=exploration_mult,
                    lambda_=reg,
                )

            vecs, rewards = gen_warmup_vecs_and_rewards(10)
            vecs, rewards = vecs.to(device), rewards.to(device)
            agent.train_dataset.set_(vecs, rewards)

            # Warmup
            for vec in vecs:
                activ, grad = agent.compute_activation_and_grad(vec)
                agent.U += grad * grad

            agent.net.train()
            agent.train(max_n_steps, patience=max_n_steps, lds=False, use_decay=True)
            agent.net.eval()

            # Playing
            for j in range(n_trials):
                agent.net.train()
                a_t, idx, best_member_grad = do_gradient_optim(
                    agent, 3 * 60, x, lr=1e-2, bounds=bounds
                )
                agent.net.eval()
                r_t = reward_fn(a_t).unsqueeze(0).unsqueeze(0)

                agent.U += best_member_grad * best_member_grad

                agent.train_dataset.add(a_t, r_t)

                agent.net.train()
                agent.train(
                    max_n_steps, patience=max_n_steps, lds=False, use_decay=True
                )
                agent.net.eval()

            # Stop using mask in evaluation
            sol, _, _ = agent.find_solution_in_vecs(x, 0, n_sigmas=n_sigmas)
            # sol = sol.to(device)
            n_sol = len(sol)

            jaccard, n_inter = compute_jaccard(sol, true_sol_idx)
            percent_found = n_inter / n_true_sol
            if n_sol == 0:
                percent_inter = 0
            else:
                percent_inter = n_inter / n_sol

            metrics_dict[algo][str(exploration_mult)]["jaccards"].append(jaccard)
            metrics_dict[algo][str(exploration_mult)]["percent_inter"].append(
                percent_inter
            )
            metrics_dict[algo][str(exploration_mult)]["percent_found"].append(
                percent_found
            )

            if n_sol == 0:
                logging.info(f"Found no solution for run {i}")
                plot_estimate(
                    agent,
                    n_trials,
                    fn=f"grad_no_sol_{algo}_decay_expl_{exploration_mult}_100_trials_seed_{i}",
                )
                metrics_dict[algo][str(exploration_mult)]["fails"] += 1

            logging.info(
                f"jaccard: {jaccard}, percent_inter: {percent_inter}, percent_found: {percent_found}"
            )

torch.save(metrics_dict, f"metrics_dict_exp_synth_{max_n_steps}_decay.pth")


# %%
logging.info("GRAD_L2")
logging.info("GRAD_L2")
logging.info("GRAD_L2")
for algo in metrics_dict.keys():
    logging.info(f"Algo: {algo}")
    for expl_mult in metrics_dict[algo].keys():
        logging.info(f"Expl mult: {expl_mult}")
        for metric in metrics_dict[algo][expl_mult].keys():
            logging.info(f"Metric: {metric}")
            logging.info(
                f"mean: {np.mean(metrics_dict[algo][expl_mult][metric])} +- {np.std(metrics_dict[algo][expl_mult][metric])} "
            )
            logging.info(
                f"interval:  [{np.min(metrics_dict[algo][expl_mult][metric])}, {np.max(metrics_dict[algo][expl_mult][metric])}]"
            )
            logging.info("============================================")
        # if 0 in metrics_dict[algo][expl_mult]:
        #     logging.info(f'rerun {metrics_dict[algo][expl_mult][metric].index(0)} with plotting')

    logging.info("============================================")


# %% [markdown]
# # Keep dropout active all the time instead of turning it off during training

# %%
metrics_dict = {}
algos = ["regTS"]
logging.info(metrics_dict)
max_n_steps = 10
for algo in algos:
    metrics_dict[algo] = {}
    for dropout_rate in [0.2, 0.5, 0.8]:
        metrics_dict[algo][str(dropout_rate)] = {}
        metrics_dict[algo][str(dropout_rate)]["jaccards"] = []
        metrics_dict[algo][str(dropout_rate)]["percent_inter"] = []
        metrics_dict[algo][str(dropout_rate)]["percent_found"] = []
        metrics_dict[algo][str(dropout_rate)]["fails"] = 0
        b = 0

        for i in range(b, 50):
            logging.info(f"at algo: {algo}, dropout: {dropout_rate}, run: {i}")
            make_deterministic(seed=i)
            vecs, rewards = gen_warmup_vecs_and_rewards(10)
            vecs, rewards = vecs.to(device), rewards.to(device)
            n_trials = 500
            width = 100
            net = NetworkDropout(d, width, dropout=dropout_rate).to(device)
            reg = 1
            sampletype = "f"
            reward_fn = reward_fn
            de_config = DEConfig
            de_policy = PullPolicy
            bern_p = 1 - dropout_rate
            p_vec = torch.tensor([bern_p] * width)

            agent = OptimNeuralTS(net, lambda_=reg, style="ts", sampletype=sampletype)

            vecs, rewards = gen_warmup_vecs_and_rewards(10)
            vecs, rewards = vecs.to(device), rewards.to(device)
            agent.train_dataset.set_(vecs, rewards)

            # Warmup
            for vec in vecs:
                activ, grad = agent.compute_activation_and_grad(vec)
                agent.U += grad * grad

                # agent.net.eval()
            agent.net.train()
            agent.train(max_n_steps, patience=max_n_steps, lds=False)

            # Train
            for j in range(n_trials):
                agent.net.train()

                a_t, idx, best_member_grad = do_gradient_optim(
                    agent, 3 * 60, x, lr=1e-2, bounds=bounds
                )
                r_t = reward_fn(a_t).unsqueeze(0).unsqueeze(0)

                if best_member_grad is None:
                    break
                agent.U += best_member_grad * best_member_grad

                agent.train_dataset.add(a_t, r_t)

                # agent.net.eval()
                agent.train(max_n_steps, patience=max_n_steps, lds=False)

            if best_member_grad is None:
                metrics_dict[algo][str(dropout_rate)]["fails"] += 1
                logging.info(
                    f"Encountered a fail in {algo} {dropout_rate} because of nans"
                )
                continue

            agent.net.train()
            sol, _, _ = agent.find_solution_in_vecs(x, 0, n_sigmas=n_sigmas)
            # sol = sol.to(device)
            n_sol = len(sol)

            jaccard, n_inter = compute_jaccard(sol, true_sol_idx)
            percent_found = n_inter / n_true_sol
            if n_sol == 0:
                percent_inter = 0
            else:
                percent_inter = n_inter / n_sol

            metrics_dict[algo][str(dropout_rate)]["jaccards"].append(jaccard)
            metrics_dict[algo][str(dropout_rate)]["percent_inter"].append(percent_inter)
            metrics_dict[algo][str(dropout_rate)]["percent_found"].append(percent_found)

            if n_sol == 0:
                logging.info(f"Found no solution for run {i}")
                metrics_dict[algo][str(dropout_rate)]["fails"] += 1

                plot_estimate(
                    agent,
                    n_trials,
                    fn=f"grad_no_sol_{algo}_drop_{dropout_rate}_100_trials_seed_{i}.png",
                )

            logging.info(
                f"jaccard: {jaccard}, percent_inter: {percent_inter}, percent_found: {percent_found}"
            )

torch.save(metrics_dict, f"metrics_dict_exp_synth_{max_n_steps}_dropout.pth")


# %%
logging.info("GRAD_DROPOUT")
logging.info("GRAD_DROPOUT")
logging.info("GRAD_DROPOUT")
for algo in metrics_dict.keys():
    logging.info(f"Algo: {algo}")
    for dropout_rate in metrics_dict[algo].keys():
        logging.info(f"dropout: {dropout_rate}")
        for metric in metrics_dict[algo][dropout_rate].keys():
            logging.info(f"Metric: {metric}")
            logging.info(
                f"mean: {np.mean(metrics_dict[algo][dropout_rate][metric])} +- {np.std(metrics_dict[algo][dropout_rate][metric])} "
            )
            logging.info(
                f"interval:  [{np.min(metrics_dict[algo][dropout_rate][metric])}, {np.max(metrics_dict[algo][dropout_rate][metric])}]"
            )
            logging.info("============================================")
        # if 0 in metrics_dict[algo][expl_mult]:
        #     logging.info(f'rerun {metrics_dict[algo][expl_mult][metric].index(0)} with plotting')

    logging.info("============================================")


# %%
