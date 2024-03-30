import numpy as np

from data.mnist import TensorDataset


def random_coreset(dataset, num_coreset_samples):
    coreset_indices = np.random.choice(np.arange(len(dataset)), num_coreset_samples, replace=False)
    train_indices = np.setdiff1d(np.arange(len(dataset)), coreset_indices)
    coreset_dataset = TensorDataset(dataset.data[coreset_indices], dataset.targets[coreset_indices])
    train_dataset = TensorDataset(dataset.data[train_indices], dataset.targets[train_indices])

    return train_dataset, coreset_dataset


def k_center_coreset(dataset, num_coreset_samples):
    # Initialize coreset with a random sample
    coreset_indices = np.random.choice(np.arange(len(dataset)), 1, replace=False)
    dataset_indices = np.setdiff1d(np.arange(len(dataset)), coreset_indices)

    while len(coreset_indices) < num_coreset_samples:
        distances = np.inf * np.ones(len(dataset_indices))
        for i, idx in enumerate(dataset_indices):
            for j in coreset_indices:
                distances[i] = min(distances[i], np.linalg.norm(dataset[idx] - dataset[j]))

        new_idx = dataset_indices[np.argmax(distances)]
        coreset_indices = np.append(coreset_indices, new_idx)
        dataset_indices = np.setdiff1d(dataset_indices, new_idx)

    coreset_dataset = TensorDataset(dataset.data[coreset_indices], dataset.targets[coreset_indices])
    train_dataset = TensorDataset(dataset.data[dataset_indices], dataset.targets[dataset_indices])
    return train_dataset, coreset_dataset