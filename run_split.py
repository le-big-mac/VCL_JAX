import sys

from jax import random
import numpy as np

from data.coreset import random_coreset
from data.mnist import get_split_MNIST
from vcl import vcl

seed = int(sys.argv[1])
print("Running split experiment")
print(f"Seed: {seed}")

key = random.PRNGKey(seed)
np.random.seed(seed)

hparams = {
    'input_size': 784,
    'hidden_size': [256, 256],
    'output_size': 2,
    'num_train_samples': 10,
    'num_pred_samples': 100,
    'num_epochs': 120,
    'batch_size': 512
}

task_train_data, task_test_data = get_split_MNIST()
coreset_size = 40
coreset_selection_fn = random_coreset

vcl(key, hparams, task_train_data, task_test_data, coreset_size, coreset_selection_fn, multi_head=True)