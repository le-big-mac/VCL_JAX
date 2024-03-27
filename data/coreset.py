import numpy as np

from torch.utils import data


def random_coreset(subset, num_coreset_samples):
    coreset_indices = np.random.choice(subset.indices, num_coreset_samples, replace=False)
    dataset_indices = np.setdiff1d(subset.indices, coreset_indices)
    return data.Subset(subset.dataset, dataset_indices), data.Subset(subset.dataset, coreset_indices)


def k_center_coreset(subset, num_coreset_samples):
    # Initialize coreset with a random sample
    coreset_indices = np.random.choice(subset.indices, 1, replace=False)
    dataset_indices = np.setdiff1d(subset.indices, coreset_indices)

    while len(coreset_indices) < num_coreset_samples:
        distances = np.inf * np.ones(len(dataset_indices))
        for i, idx in enumerate(dataset_indices):
            for j in coreset_indices:
                distances[i] = min(distances[i], np.linalg.norm(subset.dataset[idx] - subset.dataset[j]))

        new_idx = dataset_indices[np.argmax(distances)]
        coreset_indices = np.append(coreset_indices, new_idx)
        dataset_indices = np.setdiff1d(dataset_indices, new_idx)

    return data.Subset(subset.dataset, dataset_indices), data.Subset(subset.dataset, coreset_indices)