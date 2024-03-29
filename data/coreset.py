import numpy as np

from torch.utils import data


def random_coreset(subset, num_coreset_samples):
    try:
        subset_indices = subset.indices
        dataset = subset.dataset
    except AttributeError:
        subset_indices = np.arange(len(subset))
        dataset = subset

    coreset_indices = np.random.choice(subset_indices, num_coreset_samples, replace=False)
    dataset_indices = np.setdiff1d(subset_indices, coreset_indices)
    return data.Subset(dataset, dataset_indices), data.Subset(dataset, coreset_indices)


def k_center_coreset(subset, num_coreset_samples):
    try:
        subset_indices = subset.indices
        dataset = subset.dataset
    except AttributeError:
        subset_indices = np.arange(len(subset))
        dataset = subset

    # Initialize coreset with a random sample
    coreset_indices = np.random.choice(subset_indices, 1, replace=False)
    dataset_indices = np.setdiff1d(subset_indices, coreset_indices)

    while len(coreset_indices) < num_coreset_samples:
        distances = np.inf * np.ones(len(dataset_indices))
        for i, idx in enumerate(dataset_indices):
            for j in coreset_indices:
                distances[i] = min(distances[i], np.linalg.norm(dataset[idx] - dataset[j]))

        new_idx = dataset_indices[np.argmax(distances)]
        coreset_indices = np.append(coreset_indices, new_idx)
        dataset_indices = np.setdiff1d(dataset_indices, new_idx)

    return data.Subset(dataset, dataset_indices), data.Subset(dataset, coreset_indices)