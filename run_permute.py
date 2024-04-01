import sys

from jax import random
import numpy as np

from data.coreset import random_coreset
from data.mnist import get_permuted_MNIST
from vcl import vcl

seed = int(sys.argv[1])
print("Running permutation experiment")
print(f"Seed: {seed}")

key = random.PRNGKey(seed)
np.random.seed(seed)

hparams = {
    'input_size': 784,
    'hidden_size': [100, 100],
    'output_size': 10,
    'num_train_samples': 10,
    'num_pred_samples': 100,
    'num_epochs': 100,
    'batch_size': 256
}

task_train_data, task_test_data = get_permuted_MNIST(5)
coreset_size = 0
coreset_selection_fn = random_coreset

vcl(key, hparams, task_train_data, task_test_data, coreset_size, coreset_selection_fn, multi_head=False)
