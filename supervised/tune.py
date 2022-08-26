import sys

import torch
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

sys.path.append("../..")
from networks import Network, VariableNet

from fit_sup_utils import *

# using_cpu = True
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# if device == torch.device("cuda"):
# torch.set_default_tensor_type("torch.cuda.FloatTensor")
# using_cpu = False

num_cpus = len(os.sched_getaffinity(0))

logging.basicConfig(level=logging.INFO)


def run_config(config):
    n_layers = config["hidden"]
    width = config["width"]
    n_obs = config["n_obs"]
    decay = config["decay"]
    dataset = config["dataset"]
    lr = config["lr"]
    lds = config["lds"]
    batch_size = config["batch_size"]
    batch_norm = config["batch_norm"]
    patience = config["patience"]
    validation = config["validation"]
    optim_name = config["optim"]
    custom_layers = None
    noval = validation is None

    n_outputs = 1
    pred_idx = 0

    criterion = torch.nn.MSELoss()

    l = []
    for k, v in config.items():
        l += [f"{k}={v}"]

    early_stopping = EarlyStoppingActiv(patience=patience)

    make_deterministic(seed=seed)

    trainloader, training_data, X_val, y_val, n_dim, X_test, y_test = setup_data(
        dataset,
        batch_size,
        n_obs,
        lds,
        None,
        validation,
        dataset_path="/home/quo/Documents/Maitrise/optimneuralbandits/testing/datasets",
        device="cpu",
    )

    X_train, y_train = training_data.combis, training_data.labels

    if custom_layers is not None:
        net = VariableNet(n_dim, custom_layers)
    else:
        net = Network(n_dim, n_layers, n_outputs, width, None, batch_norm).to(device)

    if decay == "epoch":
        decay_val = 1
    else:
        decay_val = decay

    if lr == "plateau":
        if optim_name == "adam":
            optim = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=decay_val)
        else:
            optim = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=decay_val)
        sched = ReduceLROnPlateau(optim, "min", patience=patience // 2)
    else:
        if optim_name == "adam":
            optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=decay_val)
        else:
            optim = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=decay_val)

    for e in range(n_epochs):
        if decay == "epoch":
            optim.param_groups[0]["weight_decay"] = 1 / (e + 1)

        ### TRAIN ###
        for X, y in trainloader:
            optim.zero_grad()
            train_activ = net(X)
            train_loss = criterion(train_activ, y)
            train_loss.backward()
            optim.step()

        ### EVAL ###
        with torch.no_grad():
            net.eval()
            # Compute losses
            test_activ = net(X_test)
            test_loss = criterion(test_activ, y_test)
            session.report({"test_loss": test_loss.item()})

            val_activ = net(X_val)
            val_loss = criterion(val_activ, y_val)

            net.train()

            # Update LR scheduler
            if type(lr) == str:
                sched.step(val_loss)

        if early_stopping.early_stop:
            break


seed = 42
n_epochs = 100
make_deterministic(seed)

search_space = {
    "dataset": tune.choice(
        [
            # "50_rx_100000_combis_4_patterns_3",
            # "100_rx_100000_combis_10_patterns_35",
            "500_rx_100000_combis_10_patterns_23",
        ]
    ),
    "width": tune.choice([128]),
    "hidden": tune.choice([1]),
    "n_obs": tune.choice([100, 1000, 10000, 20000]),
    "decay": tune.choice([0.01, 0.001, 0.0001, 0, "epoch"]),
    "lr": tune.choice([0.001, 0.01, 0.1, "plateau"]),
    "lds": tune.choice([True]),
    "batch_size": tune.choice([64, 128, 256, 512, 1024]),
    "batch_norm": tune.choice([True, False]),
    "patience": tune.choice([10, 25, 50]),
    "validation": tune.choice([None, "bins", "extrema"]),
    "optim": tune.choice(["adam", "sgd"]),
}
algo = OptunaSearch()

tuner = tune.Tuner(
    run_config,
    tune_config=tune.TuneConfig(
        metric="test_loss",
        mode="min",
        search_alg=algo,
        scheduler=ASHAScheduler(),
        num_samples=1000,
    ),
    param_space=search_space,
)

results = tuner.fit()

df = results.get_dataframe()
df = df.nsmallest(10, columns="test_loss")
df.to_csv("tuner_top10.csv")
