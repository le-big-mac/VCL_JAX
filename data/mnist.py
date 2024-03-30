import numpy as np

import jax.numpy as jnp
from jax.tree_util import tree_map
import torch
from torch.utils import data
from torchvision.datasets import MNIST


class TensorDataset(data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


class SampleLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        num_samples,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
        self.num_samples = num_samples

    def collate_fn(self, batch):
        batch = tree_map(np.asarray, data.default_collate(batch))
        d, t = batch
        d = d.reshape(d.shape[0], -1)
        if self.num_samples > 1:
            t = jnp.repeat(t[jnp.newaxis, ...], self.num_samples, axis=0)
        return d, t


def get_MNIST():
    train_dataset = MNIST(
        root="data",
        train=True,
        download=True,
    )
    test_dataset = MNIST(
        root="data",
        train=False,
        download=True,
    )

    train_dataset = TensorDataset(train_dataset.data.reshape(-1, 784), train_dataset.targets)
    test_dataset = TensorDataset(test_dataset.data.reshape(-1, 784), test_dataset.targets)

    return train_dataset, test_dataset


def get_split_MNIST():
    train_dataset, test_dataset = get_MNIST()

    train_data = []
    test_data = []

    # Digit pairs: (0, 1), (2, 3), (4, 5), (6, 7), (8, 9)
    digit_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    for digit1, digit2 in digit_pairs:
        # Filter train dataset for digit1 and digit2
        train_indices = np.where((train_dataset.targets == digit1) | (train_dataset.targets == digit2))[0]
        train_subset = TensorDataset(train_dataset.data[train_indices], train_dataset.targets[train_indices] % 2)
        train_data.append(train_subset)

        # Filter test dataset for digit1 and digit2
        test_indices = np.where((test_dataset.targets == digit1) | (test_dataset.targets == digit2))[0]
        test_subset = TensorDataset(test_dataset.data[test_indices], test_dataset.targets[test_indices] % 2)
        test_data.append(test_subset)

    return train_data, test_data


def get_permuted_MNIST(num_tasks=5):
    train_dataset, test_dataset = get_MNIST()

    train_data = []
    test_data = []

    # Generate random permutations
    permutations = [np.random.permutation(784) for _ in range(num_tasks)]
    for perm in permutations:
        train_subset = TensorDataset(train_dataset.data[:, perm], train_dataset.targets)
        train_data.append(train_subset)

        test_subset = TensorDataset(test_dataset.data[:, perm], test_dataset.targets)
        test_data.append(test_subset)

    return train_data, test_data


def combine_datasets(datasets):
    data = []
    targets = []
    for dataset in datasets:
        data.append(dataset.data)
        targets.append(dataset.targets)

    data = torch.cat(data, axis=0)
    targets = torch.cat(targets, axis=0)

    return TensorDataset(data, targets)