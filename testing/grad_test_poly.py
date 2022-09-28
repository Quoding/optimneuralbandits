import json
import logging
import sys
import os

import numpy as np
import pandas as pd
import torch

sys.path.append("..")
from utils import *
from optimneuralts import DENeuralTSDiag, LenientDENeuralTSDiag
from networks import Network

using_cpu = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device("cuda"):
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    using_cpu = False

num_cpus = len(os.sched_getaffinity(0))

logging.basicConfig(level=logging.INFO)
torch.set_num_threads(num_cpus)

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
lds = args.lds
use_decay = args.usedecay
n_train = args.ntrain
train_every = args.train_every

if lds == "True" or lds == "False":
    lds = lds == "True"

make_deterministic(seed)

# combis, risks, pat_vecs, n_obs, n_dim = load_dataset(dataset, "testing/datasets")
combis, risks, pat_vecs, n_obs, n_dim = load_dataset(dataset)
init_probas = torch.tensor([1 / len(combis)] * len(combis))

reward_fn = lambda idx: (
    risks[idx],
    risks[idx],
)
# reward_fn = lambda idx: (
#     risks[idx] + torch.normal(torch.tensor([0.0]), torch.tensor([0.1])),
#     risks[idx],
# )

net = Network(
    n_dim, n_hidden_layers, n_output=1, hidden_size=width, batch_norm=batch_norm
).to(device)

#### METRICS ####
recalls = []
precisions = []
ratio_found_pats = []
recalls_alls = []
precisions_alls = []
ratio_found_pats_alls = []
n_inter_alls = []
losses = []
dataset_losses = []
all_flagged_combis_idx = set()
all_flaggeds_risks = []
all_flagged_pats_idx = set()

# Define true solution
true_sol_idx = torch.where(risks > thresh)[0]
true_sol = combis[true_sol_idx]
true_sol_idx = set(true_sol_idx.tolist())
n_combis_in_sol = len(true_sol_idx)

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

X_train, y_train, X_val, y_val = get_data_splits(vecs, rewards, valtype=valtype)

agent.train_dataset.set_(X_train, y_train)
if valtype != "noval":
    agent.val_dataset.set_(X_val, y_val)


logging.info("Warming up...")
#### WARMUP ####
agent.net.eval()
for vec in agent.train_dataset.features:
    activ, grad = agent.compute_activation_and_grad(vec[None])
    agent.U += grad * grad

agent.net.train()
loss = agent.train(
    n_epochs,
    lr=lr,
    batch_size=batch_size,
    patience=patience,
    lds=lds,
    n_train=n_train,
    use_decay=use_decay,
)
agent.net.eval()


logging.info("Warm up over. Computing metrics...")

## VISUALIZE REPRESENTATION AFTER WARMUP ###
# import matplotlib.pyplot as plt

# with torch.no_grad():
#     plt.scatter(risks.cpu().numpy(), agent.net(combis).cpu().numpy())
#     plt.plot([0, 4], [0, 4], color="black", linestyle="dashed")
#     plt.xlim(0, 4)
#     plt.ylim(0, 4)
#     plt.show()

## GET METRICS POST WARMUP, PRE TRAINING ####
(
    recall,
    precision,
    percent_found_pat,
    n_inter,
    recall_all,
    precision_all,
    percent_found_pat_all,
    n_inter_all,
    all_flagged_combis_idx,
    all_flagged_pats_idx,
) = compute_metrics(
    agent,
    combis,
    thresh,
    pat_vecs,
    true_sol_idx,
    n_sigmas,
    all_flagged_combis_idx,
    all_flagged_pats_idx,
)
logging.info(
    f"recall: {recall}, precision: {precision}, ratio of patterns found: {percent_found_pat}, n_inter: {n_inter}"
)
logging.info(
    f"recall all: {recall_all}, precision all: {precision_all}, ratio of patterns found all: {percent_found_pat_all}, n_inter all: {n_inter_all}"
)
recalls.append(recall)
precisions.append(precision)
ratio_found_pats.append(percent_found_pat)
recalls_alls.append(recall_all)
precisions_alls.append(precision_all)
ratio_found_pats_alls.append(percent_found_pat_all)
n_inter_alls.append(n_inter_all)
losses.append(loss)
logging.info("Post warmup metrics over. Starting training")


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

    if (i + 1) % train_every == 0:
        agent.net.train()
        loss = agent.train(
            n_epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            lds=lds,
            n_train=n_train,
            use_decay=use_decay,
        )
        agent.net.eval()
    #### COMPUTE METRICS ####
    if (i + 1) % 200 == 0:
        (
            recall,
            precision,
            percent_found_pat,
            n_inter,
            recall_all,
            precision_all,
            percent_found_pat_all,
            n_inter_all,
            all_flagged_combis_idx,
            all_flagged_pats_idx,
        ) = compute_metrics(
            agent,
            combis,
            thresh,
            pat_vecs,
            true_sol_idx,
            n_sigmas,
            all_flagged_combis_idx,
            all_flagged_pats_idx,
        )

        with torch.no_grad():
            dataset_activ = agent.net(combis)
            dataset_loss = agent.loss_func(dataset_activ, risks)
            dataset_losses.append(dataset_loss.item())

        recalls.append(recall)
        precisions.append(precision)
        ratio_found_pats.append(percent_found_pat)
        recalls_alls.append(recall_all)
        precisions_alls.append(precision_all)
        ratio_found_pats_alls.append(percent_found_pat_all)
        n_inter_alls.append(n_inter_all)

        losses.append(loss)

        logging.info(
            f"trial: {i + 1}, recall: {recall}, precision: {precision}, ratio of patterns found: {percent_found_pat}, n_inter: {n_inter}, loss: {loss}, dataset_loss: {dataset_loss}"
        )
        logging.info(
            f"recall all: {recall_all}, precision all: {precision_all}, ratio of patterns found all: {percent_found_pat_all}, n_inter all: {n_inter_all}"
        )


output_dir = args.output
l = [
    "agents",
    "recalls",
    "precisions",
    "ratio_found_pats",
    "losses",
    "dataset_losses",
    "recalls_alls",
    "precisions_alls",
    "ratio_found_pats_alls",
    "n_inter_alls",
    "all_flagged_combis_idx",
    "all_flagged_risks",
]

try:
    all_flagged_risks = risks[torch.tensor(list(all_flagged_combis_idx))]
except IndexError as e:
    logging.info("No flagged combination during the entire experiment")
    logging.info("all_flagged_risks is now an empty tensor")
    all_flagged_risks = torch.tensor([])

for item in l:
    os.makedirs(f"{output_dir}/{item}/", exist_ok=True)

torch.save(agent, f"{output_dir}/agents/{seed}.pth")
torch.save(recalls, f"{output_dir}/recalls/{seed}.pth")
torch.save(precisions, f"{output_dir}/precisions/{seed}.pth")
torch.save(ratio_found_pats, f"{output_dir}/ratio_found_pats/{seed}.pth")
torch.save(losses, f"{output_dir}/losses/{seed}.pth")
torch.save(dataset_losses, f"{output_dir}/dataset_losses/{seed}.pth")
torch.save(recalls_alls, f"{output_dir}/recalls_alls/{seed}.pth")
torch.save(precisions_alls, f"{output_dir}/precisions_alls/{seed}.pth")
torch.save(ratio_found_pats_alls, f"{output_dir}/ratio_found_pats_alls/{seed}.pth")
torch.save(n_inter_alls, f"{output_dir}/n_inter_alls/{seed}.pth")
torch.save(all_flagged_risks, f"{output_dir}/all_flagged_risks/{seed}.pth")
torch.save(all_flagged_combis_idx, f"{output_dir}/all_flagged_combis_idx/{seed}.pth")
