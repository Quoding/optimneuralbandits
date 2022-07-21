import json
import logging
import sys
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from detorch import Strategy

sys.path.append("..")
from utils import *
from optimneuralts import DENeuralTSDiag, LenientDENeuralTSDiag
from networks import Network


#### SET UP ####
args = parse_args()


#### PARAMETERS ####
n_trials = args.trials
dataset = args.dataset
thresh = args.threshold
seed = args.seed
width = args.width
n_hidden_layers = args.layers
reg = args.reg
exploration_mult = args.exploration
n_epochs = args.n_epochs
lr = args.lr
style = args.style
optim_string = args.optimizer
n_warmup = args.warmup
pop_optim_n_members = args.pop_n_members
pop_optim_n_steps = args.pop_n_steps
pop_optim_lr = args.pop_lr
batch_size = args.batch_size
n_sigmas = args.n_sigmas
ci_thresh = args.ci_thresh
patience = args.patience
valtype = args.valtype
batch_norm = not args.nobatchnorm
use_lds = not args.nolds


make_deterministic(seed)

combis, risks, pat_vecs, n_obs, n_dim = load_dataset(dataset, "testing/datasets")

init_probas = torch.tensor([1 / len(combis)] * len(combis))

reward_fn = lambda idx: (
    risks[idx] + torch.normal(torch.tensor([0.0]), torch.tensor([0.1])),
    risks[idx],
)


class DEConfig:
    n_step: int = pop_optim_n_steps
    population_size: int = pop_optim_n_members
    differential_weight: float = 1
    crossover_probability: float = 0.9
    strategy: Strategy = Strategy.best1bin
    seed: int = "doesn't matter"


#### SET UP NETWORK AND DE ####
de_config = DEConfig
de_policy = PullPolicy
net = Network(n_dim, n_hidden_layers, hidden_size=width, batch_norm=batch_norm).to(
    device
)

#### METRICS ####
jaccards = []
ratio_apps = []
percent_found_pats = []
losses = []
dataset_losses = []

# Define true solution
combis_in_sol = torch.where(risks > thresh)[0]
true_sol = combis[combis_in_sol]
n_combis_in_sol = len(combis_in_sol)

logging.info(f"There are {n_combis_in_sol} combinations in the solution set")

agent = DENeuralTSDiag(
    net,
    optim_string,
    nu=exploration_mult,
    lambda_=reg,
    style=style,
    valtype=valtype,
)

vecs, rewards = gen_warmup_vecs_and_rewards(n_warmup, combis, risks, init_probas)

X_train, y_train, X_val, y_val = get_data_splits(vecs, rewards, val=valtype)

agent.train_dataset.set_(X_train, y_train)
agent.val_dataset.set_(X_val, y_val)

logging.info("Warming up...")
#### WARMUP ####
agent.train(n_epochs, lr=lr, batch_size=batch_size, patience=patience, use_lds=use_lds)

## GET METRICS POST WARMUP, PRE TRAINING ####
# jaccard, ratio_app, percent_found_pat, n_inter = compute_metrics(
#     agent, combis, thresh, pat_vecs, true_sol, n_sigmas
# )
# logging.info(
#     f"jaccard: {jaccard}, ratio_app: {ratio_app}, ratio of patterns found: {percent_found_pat}, n_inter: {n_inter}"
# )
# jaccards.append(jaccard)
# ratio_apps.append(ratio_app)
# percent_found_pats.append(percent_found_pat)
logging.info("Warm up over. Starting training")

#### TRAINING ####
for i in range(n_trials):
    a_t, idx, best_member_grad = do_gradient_optim(
        agent, pop_optim_n_members, combis, lr=pop_optim_lr
    )
    r_t, true_r = reward_fn(idx)
    r_t = r_t[:, None]
    agent.U += best_member_grad * best_member_grad

    a_train, r_train = agent.val_dataset.update(a_t, r_t)
    agent.train_dataset.add(a_train, r_train)

    loss = agent.train(
        n_epochs, lr=lr, batch_size=batch_size, patience=patience, use_lds=use_lds
    )
    print(i)
    #### COMPUTE METRICS ####
    if (i + 1) % 100 == 0:
        jaccard, ratio_app, percent_found_pat, n_inter = compute_metrics(
            agent, combis, thresh, pat_vecs, true_sol, n_sigmas
        )

        with torch.no_grad():
            dataset_activ = agent.net(combis)
            dataset_loss = agent.loss_func(dataset_activ, risks)
            dataset_losses.append(dataset_loss.item())

        jaccards.append(jaccard)
        ratio_apps.append(ratio_app)
        percent_found_pats.append(percent_found_pat)
        losses.append(loss)

        logging.info(
            f"trial: {i + 1}, jaccard: {jaccard}, ratio_app: {ratio_app}, ratio of patterns found: {percent_found_pat}, n_inter: {n_inter}, loss: {loss}, dataset_loss: {dataset_loss}"
        )

output_dir = args.output
l = ["agents", "jaccards", "ratio_apps", "ratio_found_pats", "losses", "dataset_losses"]
for item in l:
    os.makedirs(f"{output_dir}/{item}/", exist_ok=True)

torch.save(agent, f"{output_dir}/agents/{seed}.pth")
torch.save(jaccards, f"{output_dir}/jaccards/{seed}.pth")
torch.save(ratio_apps, f"{output_dir}/ratio_apps/{seed}.pth")
torch.save(percent_found_pats, f"{output_dir}/ratio_found_pats/{seed}.pth")
torch.save(losses, f"{output_dir}/losses/{seed}.pth")
torch.save(dataset_losses, f"{output_dir}/dataset_losses/{seed}.pth")

# TODO Negative of sample_r should give proper gradients and not 0.
