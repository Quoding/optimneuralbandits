import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import conv1d
from utils import discretize_targets, build_histogram, gaussian_fn


class ReplayDataset(Dataset):
    def __init__(self, features=None, rewards=None):
        self.features = features
        self.rewards = rewards

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        return self.features[idx], self.rewards[idx]

    def set_(self, features_2d, rewards_2d):
        self.features = features_2d
        self.rewards = rewards_2d

    def add(self, features, reward):
        if features is None or reward is None:
            return
        self.features = torch.cat((self.features, features))
        self.rewards = torch.cat((self.rewards, reward))

    def get_weights(self, kern_size=5, kern_sigma=2, reweight="sqrt_inv"):
        bin_size = 0.1
        factor = 10
        # Implements label distribution smoothing (Delving into Deep Imbalanced Regression, https://arxiv.org/abs/2102.09554)
        # Discretize the risks (labels used later)
        flat_labels = self.rewards.flatten()
        discrete_risks = discretize_targets(flat_labels, factor)

        hist, n_bins, list_bin_edges = build_histogram(flat_labels, factor, bin_size)
        weights = hist.hist

        # wplot = weights.cpu().numpy()
        # plt.bar(list_bin_edges[:-1], wplot, width=0.1, edgecolor="black")
        # plt.title("Distribution des risques relatifs (discret)")
        # plt.xlabel("Intervalles (0.1)")
        # plt.ylabel("Compte")
        # plt.show()

        if reweight == "sqrt_inv":
            weights = torch.sqrt(weights)

        # Apply label distribution smoothing with gaussian filter
        # Get the gaussian filter
        kernel = gaussian_fn(kern_size, kern_sigma)[None, None]
        weights = conv1d(weights[None, None].cuda(), kernel, padding=(kern_size // 2))
        weights = 1 / weights

        # Get weights for dataset
        weight_bins = {list_bin_edges[i]: weights[0][0][i] for i in range(n_bins)}
        # print(weight_bins)
        weights_per_obs = [weight_bins[risk] for risk in discrete_risks]

        # k = list(weight_bins.keys())
        # v = list(weight_bins.values())
        # v = [val.item() for val in v]

        # plt.bar(k, v, width=0.1, edgecolor="black")
        # plt.title("Poids d'echantillonnage des observations")
        # plt.xlabel("Intervalles (0.1)")
        # plt.ylabel("Poids")
        # plt.show()
        return weights_per_obs


class ValidationReplayDataset(ReplayDataset):
    def __init__(self, features=None, rewards=None, valtype=None):
        self.features = None
        self.rewards = None
        self.valtype = valtype
        self.bins_features = {}
        self.bins_rewards = {}

        if features is None or rewards is None:
            pass
        else:
            self.set_(features, rewards)

    def build_mapping(self):
        for i in range(len(self.rewards)):
            reward = self.rewards[i].unsqueeze(0)
            vec = self.features[i].unsqueeze(0)

            bin_ = int(reward * 10) / 10

            self.bins_features[bin_] = vec
            self.bins_rewards[bin_] = reward

    def __len__(self):
        return len(self.rewards)

    def set_(self, features_2d, rewards_2d):
        self.features = features_2d
        self.rewards = rewards_2d

        if self.valtype == "bins":
            self.build_mapping()

    def update(self, features, reward):
        to_training = (features, reward)
        if self.valtype == "extrema":
            # If valtype is extrema, then the set is composed of only 2 observations, the min and max of rewards
            # Assumes sorted tensors based on self.rewards
            if reward < min(self.rewards):
                to_training = (
                    self.features[0].clone().unsqueeze(0),
                    self.rewards[0].clone().unsqueeze(0),
                )
                self.features[0] = features[0]
                self.rewards[0] = reward[0]
            elif reward > max(self.rewards):
                to_training = (
                    self.features[1].clone().unsqueeze(0),
                    self.rewards[1].clone().unsqueeze(0),
                )
                self.features[1] = features[0]
                self.rewards[1] = reward[0]

        elif self.valtype == "bins":
            # Update the bins with the new observation
            new_obs_bin = int(reward * 10) / 10
            # Send the observation that was already in the bin to the training set
            to_training = (
                self.bins_features.get(new_obs_bin, None),
                self.bins_rewards.get(new_obs_bin, None),
            )
            # Update the bin
            self.bins_features[new_obs_bin] = features
            self.bins_rewards[new_obs_bin] = reward

            self.rewards = torch.cat(list(self.bins_rewards.values()))
            self.features = torch.cat(list(self.bins_features.values()))

        return to_training
