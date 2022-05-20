# %%
import json
import logging
import sys
import os

import numpy as np
import pandas as pd
import torch

from utils import *

sys.path.append("..")
from optimneuralts import DENeuralTSDiag, LenientDENeuralTSDiag


#### SET UP ####
args = parse_args()

combis, risks, patterns, n_obs, n_dim = load_dataset(args.dataset)
pat_vecs = torch.tensor(
    [patterns[f"pattern_{i}"]["pattern"] for i in range(len(patterns))]
)
combis, risks = (
    torch.tensor(combis.values).float(),
    torch.tensor(risks.values).unsqueeze(1).float(),
)

init_probas = torch.tensor([1 / len(combis)] * len(combis))

#### PARAMETERS ####
seed = args.seed
make_deterministic(seed)
thresh = args.threshold
n_trials = args.trials
width = args.width
n_hidden_layers = args.layers
reg = args.reg
exploration_mult = args.exploration
reward_fn = lambda idx: risks[idx]
max_n_steps = args.n_optim_steps
style = args.style
lr = args.lr
n_warmup = args.warmup
optim_string = args.optimizer


#### SET UP NETWORK AND DE ####
de_config = DEConfig
de_policy = PullPolicy
net = Network(n_dim, n_hidden_layers, width).to(device)

#### METRICS ####
hist_solution = []
hist_solution_pat = []
jaccards = []
ratio_apps = []
percent_found_pats = []
losses = []
dataset_losses = []

# Define true solution
combis_in_sol = torch.where(risks >= thresh)[0]
true_sol = combis[combis_in_sol]
n_combis_in_sol = len(combis_in_sol)

logging.info(f"There are {n_combis_in_sol} combinations in solution set")

agent = DENeuralTSDiag(net, optim_string, nu=exploration_mult, lamdba=reg, style=style)
vecs, rewards = gen_warmup_vecs_and_rewards(n_warmup, combis, risks, init_probas)

logging.info("Warming up...")
#### WARMUP ####
for i in range(len(rewards)):
    agent.vec_history = vecs[: i + 1]
    agent.reward_history = rewards[: i + 1]
    vec = vecs[i]
    activ, grad = agent.compute_activation_and_grad(vec)
    agent.U += grad * grad
    agent.train(min(i + 1, max_n_steps), lr)


#### GET METRICS POST WARMUP, PRE TRAINING ####
jaccard, ratio_app, percent_found_pat, n_inter = compute_metrics(
    agent, combis, thresh, pat_vecs, true_sol
)
logging.info(
    f"jaccard: {jaccard}, ratio_app: {ratio_app}, ratio of patterns found: {percent_found_pat}, n_inter: {n_inter}"
)
jaccards.append(jaccard)
ratio_apps.append(ratio_app)
percent_found_pats.append(percent_found_pat)
logging.info("Warm up over. Starting training")

#### TRAINING ####
for i in range(n_trials):
    best_member = find_best_member(agent.get_sample, de_config, init_probas, combis, i)
    best_member_grad = best_member.activation_grad
    a_t = best_member.params.data
    a_t, idx = change_to_closest_existing_vector(a_t, combis)
    r_t = reward_fn(idx)[:, None]
    a_t = a_t[None, :]
    agent.U += best_member_grad * best_member_grad

    agent.vec_history = torch.cat((agent.vec_history, a_t))
    agent.reward_history = torch.cat((agent.reward_history, r_t))

    n_steps = min(agent.reward_history.shape[0], max_n_steps)
    loss = agent.train(n_steps, lr)

    #### COMPUTE METRICS ####
    if (i + 1) % 100 == 0:
        jaccard, ratio_app, percent_found_pat, n_inter = compute_metrics(
            agent, combis, thresh, pat_vecs, true_sol
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
