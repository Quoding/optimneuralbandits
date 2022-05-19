# %%
import sys
import numpy as np
import pandas as pd
import json
import torch
import logging

from utils import *

sys.path.append("..")
from optimneuralts import DENeuralTSDiag, LenientDENeuralTSDiag

# %%
torch.set_default_tensor_type("torch.cuda.FloatTensor")

logging.basicConfig(level=logging.INFO)

n_dim = int(sys.argv[1])
n_trials = int(sys.argv[2])
run_number = int(sys.argv[3])

epsilon = 0.4
inv_eps = 1 - epsilon


# %%
base = f"10000r_{n_dim}c_1o_run{run_number}"
logging.info(f"Loading up this dataset: {base}")

df = pd.read_csv(f"datasets/polypharmacie/{base}.csv")
with open(
    f"datasets/polypharmacie/{base}_config.json",
    "r",
) as file:
    config = json.load(file)

with open(f"datasets/polypharmacie/{base}.json", "r") as file:
    pattern_config = json.load(file)

pattern_codes = [
    np.array(pattern_config[f"pattern_{i}"]["code_indices"]) - 1 for i in range(10)
]

n_medical_codes = config["n_medical_codes"]
n_outcomes = config["n_outcomes"]
column_names = (
    ["patient_id"]
    + [f"medical_code_{i}" for i in range(n_medical_codes)]
    + [f"outcome_code_{i}" for i in range(n_outcomes)]
)

df.columns = column_names

X_df = df.iloc[:, 1 : n_medical_codes + 1]
X = X_df.values

y_df = df.iloc[:, n_medical_codes + 1 : n_medical_codes + 1 + n_outcomes]
y = y_df.values.ravel()

X_df.describe()
X = torch.Tensor(X)
y = torch.Tensor(y)
set_existing_vecs = torch.unique(X, dim=0)

logging.info(y.shape)
logging.info(X.shape)

p = torch.tensor([1 / len(set_existing_vecs)] * len(set_existing_vecs))

make_deterministic(run_number)

# %%
risks = []
for vec in set_existing_vecs:
    risks.append(risk_reward_fn(vec, X, y))
risks = np.array(risks)
max_risk = max(risks)
thresh = inv_eps * max_risk
true_sol = set_existing_vecs[np.where(risks >= thresh)]
n_true_sol = len(true_sol)
logging.info(len(set_existing_vecs))
logging.info(len(true_sol))

# Find how many pattern vecs are actually in solution
pattern_vecs_in_sol = []
pattern_risks = []
vecs_with_patterns = []
for code_indices in pattern_codes:
    vec = np.zeros(n_dim)
    vec[code_indices] = 1
    vec = torch.tensor(vec)
    pattern_risk = risk_reward_fn(vec, X, y)
    if pattern_risk >= thresh:
        pattern_vecs_in_sol.append(vec)
        for existing_vec in true_sol:
            if (existing_vec[code_indices] == 1).all():
                vecs_with_patterns.append(existing_vec)

pattern_vecs_in_sol = torch.stack(pattern_vecs_in_sol).float()
vecs_with_patterns = torch.stack(vecs_with_patterns)
vecs_with_patterns = torch.unique(vecs_with_patterns, dim=0).float()

n_patterns_in_sol = len(pattern_vecs_in_sol)

logging.info(len(pattern_vecs_in_sol))
logging.info(len(vecs_with_patterns))

# %%
width = 100
n_hidden_layers = 1
net = Network(n_dim, n_hidden_layers, width).to(device)
reg = 1
exploration_mult = 1
delay = 0
reward_fn = risk_reward_fn
de_config = DEConfig
de_policy = PullPolicy
max_n_steps = 100
lr = 1e-2

agent = DENeuralTSDiag(net, nu=exploration_mult, lamdba=reg, style="ts")

vecs, rewards = gen_warmup_vecs_and_rewards(100, X, y, p, set_existing_vecs)

for i in range(len(rewards)):
    agent.vec_history = vecs[: i + 1]
    agent.reward_history = rewards[: i + 1]
    vec = vecs[i]
    activ, grad = agent.compute_activation_and_grad(vec)
    agent.U += grad * grad
    agent.train(min(i + 1, max_n_steps), lr)

# %%
hist_solution = []
hist_solution_pat = []
jaccards = []
ratio_apps = []
percent_found_pats = []
percent_found_existing_vecs_with_pats = []

sol = agent.find_solution_in_vecs(
    set_existing_vecs, thresh
)  # Parmis tous les vecteurs existant, lesquels je trouve ? (Jaccard, ratio_app)
sol_pat = agent.find_solution_in_vecs(
    pattern_vecs_in_sol, thresh
)  # Parmis les patrons insérés, combien j'en trouve tels quels (Ratio_p.t.)
sol_existing_vecs_with_pat = agent.find_solution_in_vecs(
    vecs_with_patterns, thresh
)  # parmis les combi qui existent dans Dataset ET ont le patron, combien je trouve (Ratio de combi ayant la patron trouvées)

jaccard, n_inter = compute_jaccard(
    sol, true_sol
)  # À quel point ma solution trouvée parmis les vecteurs du dataset est similaire à la vraie solution

percent_found_pat = len(sol_pat) / len(
    pattern_vecs_in_sol
)  # Combien de patrons tels quels j'ai flag ?
percent_found_existing_vecs_with_pat = len(sol_existing_vecs_with_pat) / len(
    vecs_with_patterns
)  # Combien de combi avec un patron dans le dataset j'ai flag?
if len(sol) == 0:
    ratio_app = 0
else:
    ratio_app = n_inter / len(
        sol
    )  # A quel point ma solution trouvee parmis les vecteurs du dataset est dans la vraie solution

jaccards.append(jaccard)
ratio_apps.append(ratio_app)
percent_found_pats.append(percent_found_pat)
percent_found_existing_vecs_with_pats.append(percent_found_existing_vecs_with_pat)


logging.info(
    f"jaccard: {jaccard}, ratio_app: {ratio_app}, ratio of patterns found: {percent_found_pat}, ratio of existing vecs with pattern found: {percent_found_existing_vecs_with_pat}, n_inter: {n_inter}"
)

# %% [markdown]
# ## train

# %%

losses = []

for i in range(n_trials):
    best_member = find_best_member(agent.get_sample, de_config, p, set_existing_vecs)
    best_member_grad = best_member.activation_grad
    a_t = best_member.params.data
    a_t = change_to_closest_existing_vector(a_t, set_existing_vecs)
    r_t = torch.tensor([reward_fn(a_t, X, y)]).unsqueeze(0)
    a_t = a_t[None, :]

    agent.U += best_member_grad * best_member_grad

    agent.vec_history = torch.cat((agent.vec_history, a_t))
    agent.reward_history = torch.cat((agent.reward_history, r_t))

    n_steps = min(agent.reward_history.shape[0], max_n_steps)
    loss = agent.train(n_steps, lr)

    if (i + 1) % 100 == 0:
        logging.info(f"trial: {i + 1}")

        sol = agent.find_solution_in_vecs(
            set_existing_vecs, thresh
        )  # Parmis tous les vecteurs existant, lesquels je trouve ? (Jaccard, ratio_app)
        sol_pat = agent.find_solution_in_vecs(
            pattern_vecs_in_sol, thresh
        )  # Parmis les patrons insérés, combien j'en trouve tels quels (Ratio_p.t.)
        sol_existing_vecs_with_pat = agent.find_solution_in_vecs(
            vecs_with_patterns, thresh
        )  # parmis les combi qui existent dans Dataset ET ont le patron, combien je trouve (Ratio de combi ayant le patron trouvées)

        jaccard, n_inter = compute_jaccard(
            sol, true_sol
        )  # À quel point ma solution trouvée parmis les vecteurs du dataset est similaire à la vraie solution

        percent_found_pat = len(sol_pat) / len(
            pattern_vecs_in_sol
        )  # Combien de patrons tels quels j'ai flag ?
        percent_found_existing_vecs_with_pat = len(sol_existing_vecs_with_pat) / len(
            vecs_with_patterns
        )  # Combien de combi avec un patron dans le dataset j'ai flag?
        if len(sol) == 0:
            ratio_app = 0
        else:
            ratio_app = n_inter / len(
                sol
            )  # A quel point ma solution trouvee parmis les vecteurs du dataset est dans la vraie solution

        jaccards.append(jaccard)
        ratio_apps.append(ratio_app)
        percent_found_pats.append(percent_found_pat)
        percent_found_existing_vecs_with_pats.append(
            percent_found_existing_vecs_with_pat
        )
        losses.append(loss)

        logging.info(
            f"jaccard: {jaccard}, ratio_app: {ratio_app}, ratio of patterns found: {percent_found_pat}, ratio of existing vecs with pattern found: {percent_found_existing_vecs_with_pat}, n_inter: {n_inter}, loss: {loss}"
        )


# %%
path = "/home/alarouch/alarouch/neuralbandits/testing/saves"
# path = "saves"
append = f"d{n_dim}_trials{n_trials}_nlayers{n_hidden_layers}_max_n_steps_{max_n_steps}_run{run_number}"
torch.save(agent, f"{path}/agents/{append}.pth")
torch.save(jaccards, f"{path}/jaccards/{append}.pth")
torch.save(ratio_apps, f"{path}/ratio_apps/{append}.pth")
torch.save(percent_found_pats, f"{path}/ratio_found_pats/{append}.pth")
torch.save(
    percent_found_existing_vecs_with_pats,
    f"{path}/ratio_found_existing_vecs_with_pats/{append}.pth",
)
torch.save(losses, f"{path}/losses/{append}.pth")
