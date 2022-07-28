import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import conv1d
from utils import discretize_targets, build_histogram, gaussian_fn, device


class ReplayDataset(Dataset):
    def __init__(self, features=None, rewards=None):
        # Keep original in order to do LDS
        self.original_features = features
        self.original_rewards = rewards

        # Actually used for training purposes, may be different from original avec LDS.
        self.features = features
        self.rewards = rewards

        # Weights tensor for LDS
        self.weights_per_obs = None

    def __len__(self):
        return len(self.original_rewards)

    def __getitem__(self, idx):
        return self.original_features[idx], self.original_rewards[idx]

    def set_(self, features_2d, rewards_2d):
        self.original_features = features_2d
        self.original_rewards = rewards_2d

    def add(self, features, reward):
        if features is None or reward is None:
            return
        self.original_features = torch.cat((self.original_features, features))
        self.original_rewards = torch.cat((self.original_rewards, reward))

    def update_weights(self, kern_size=5, kern_sigma=2, reweight="sqrt_inv"):
        """Compute weights for label distribution smoothing via a gaussian kernel

        Args:
            kern_size (int, optional): Gaussian kernel size. Defaults to 5.
            kern_sigma (int, optional): Gaussian kernel sigma. Defaults to 2.
            reweight (str, optional): Type of reweighting done, must be either "sqrt_inv" or True. Defaults to "sqrt_inv".

        Returns:
            torch.Tensor (1D): sampling weights for label distribution smoothing
        """
        assert reweight in ["sqrt_inv", True]
        # Implements label distribution smoothing (Delving into Deep Imbalanced Regression, https://arxiv.org/abs/2102.09554)
        bin_size = 0.1
        factor = 10
        # Discretize the risks (labels used later)

        flat_labels = self.original_rewards.flatten()
        discrete_risks = discretize_targets(flat_labels, factor)

        hist, n_bins, list_bin_edges = build_histogram(flat_labels, factor, bin_size)
        weights = hist.hist.to(device)

        if reweight == "sqrt_inv":
            weights = torch.sqrt(weights)

        # Apply label distribution smoothing with gaussian filter
        # Get the gaussian filter
        kernel = gaussian_fn(kern_size, kern_sigma)[None, None]
        weights = conv1d(weights[None, None], kernel, padding=(kern_size // 2))
        weights = 1 / weights

        # Get weights for dataset
        weight_bins = {list_bin_edges[i]: weights[0][0][i] for i in range(n_bins)}

        # This isn't slow, looping is fast, dictionaries are hashmaps
        weights_per_obs = torch.tensor([weight_bins[risk] for risk in discrete_risks])
        self.weights_per_obs = weights_per_obs / weights_per_obs.sum()

    def update_dataset(self):
        """
        Attempt at making a fully tabular dataset that can be batched via slicing instead of individual indexing
        """
        # If we do not do LDS
        if self.weights_per_obs is None:
            # Keep original dataset as the training dataset
            self.features = self.original_features
            self.rewards = self.original_rewards
        # If we do LDS
        else:
            # Sample according to LDs weights. Make that our new training dataset.
            sampled_idx = self.weights_per_obs.multinomial(
                num_samples=len(self), replacement=True
            )
            self.features = self.original_features[sampled_idx]
            self.rewards = self.original_rewards[sampled_idx]


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
        # Set to device right away because we don't use these datapoints in a dataloader in anyway. Doing this avoids repeatedly sending to device
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
            elif reward.item() < min(self.rewards).item():
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


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        """Shuffles the dataset if needed and resets the index

        Returns:
            self
        """
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        """Picks and returns a batch

        Raises:
            StopIteration: if end of dataset reached

        Returns:
            torch.Tensor: batch of data
        """
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
