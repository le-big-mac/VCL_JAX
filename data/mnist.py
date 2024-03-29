import numpy as np

import jax.numpy as jnp
from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
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

    def collate_fn(self, batch):
        return tree_map(np.asarray, data.default_collate(batch))


class SplitLoader(NumpyLoader):
    def __init__(
        self,
        dataset,
        num_samples,
        batch_size=1,
        shuffle=False
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.num_samples = num_samples

    def collate_fn(self, batch):
        batch = super().collate_fn(batch)
        d, t = batch
        t = jnp.repeat(t[jnp.newaxis, ...], self.num_samples, axis=0)
        return d, jnp.mod(t, 2)


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


def get_split_MNIST():
    train_dataset = MNIST(
        root="data",
        train=True,
        transform=FlattenAndCast(),
        download=True,
    )
    test_dataset = MNIST(
        root="data",
        train=False,
        transform=FlattenAndCast(),
        download=True,
    )

    train_data = []
    test_data = []

    # Digit pairs: (0, 1), (2, 3), (4, 5), (6, 7), (8, 9)
    digit_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    for digit1, digit2 in digit_pairs:
        # Filter train dataset for digit1 and digit2
        train_indices = np.where((train_dataset.targets == digit1) | (train_dataset.targets == digit2))[0]
        train_subset = data.Subset(train_dataset, train_indices)
        train_data.append(train_subset)

        # Filter test dataset for digit1 and digit2
        test_indices = np.where((test_dataset.targets == digit1) | (test_dataset.targets == digit2))[0]
        test_subset = data.Subset(test_dataset, test_indices)
        test_data.append(test_subset)

    return train_data, test_data
