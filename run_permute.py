from copy import deepcopy
import sys

import jax
from jax import random
import jax.numpy as jnp
import numpy as np

from data.coreset import random_coreset
from data.mnist import get_MNIST, PermutedLoader
from models.mlp import MFVI_NN, extract_means_and_logvars
from training.train_vcl import create_train_state, train_Dt, eval_Dt

print("Running permutation experiment")

seed = int(sys.argv[1])
key = random.PRNGKey(seed)
np.random.seed(seed)
input_size = 784
hidden_size = [256, 256]
output_size = 10
num_train_samples = 10
num_pred_samples = 100
num_epochs = 120
num_tasks = 5

sizes = [input_size] + hidden_size
key, kernel_keys, bias_keys = random.split(key, 3)
kernel_keys = random.split(kernel_keys, len(sizes) - 1)
bias_keys = random.split(bias_keys, len(sizes) - 1)

truncated_normal = jax.nn.initializers.truncated_normal(stddev=0.1)
prev_hidden_means = (
    [truncated_normal(kernel_keys[i], (sizes[i], sizes[i+1]), jnp.float32) for i in range(len(sizes) - 1)],
    [truncated_normal(bias_keys[i], (sizes[i+1],), jnp.float32) for i in range(len(sizes) - 1)]
    )
prev_hidden_logvars = (
    [jnp.full((din, dout), -6.) for din, dout in zip(sizes[:-1], sizes[1:])],
    [jnp.full(dout, -6.) for dout in sizes[1:]]
    )
key, kernel_key, bias_key = random.split(key, 3)
prev_last_means = (
    [truncated_normal(kernel_key, (hidden_size[-1], output_size), jnp.float32)],
    [truncated_normal(bias_key, (output_size,), jnp.float32)]
    )
prev_last_logvars = (
    [jnp.full((hidden_size[-1], output_size), -6.)],
    [jnp.full(output_size, -6.)]
    )

train_data, test_data = get_MNIST()
permutations = [np.random.permutation(784) for _ in range(num_tasks)]

coreset_selection_fn = random_coreset
coreset_size = 40
coresets = []

for task_idx in range(num_tasks):
    train_data, coreset_data = coreset_selection_fn(train_data, coreset_size)

    train_loader = PermutedLoader(train_data, num_samples=num_train_samples, permutation=permutations[task_idx], batch_size=32, shuffle=True)
    coreset_loader = PermutedLoader(coreset_data, num_samples=num_train_samples, permutation=permutations[task_idx], batch_size=32, shuffle=True)
    coresets.append(coreset_loader)

    model = MFVI_NN(hidden_size, output_size, prev_hidden_means,
                    prev_hidden_logvars, prev_last_means,
                    prev_last_logvars,
                    num_train_samples=num_train_samples,
                    num_pred_samples=num_pred_samples)
    dummy_input = jnp.ones([1, 784])
    key, params_key = random.split(key)
    params = model.init({'params': params_key}, dummy_input, task_idx=0)["params"]
    prev_params = deepcopy(params)
    state = create_train_state(model, params, learning_rate=1e-3)

    key, subkey = random.split(key)
    state = train_Dt(subkey, state, task_idx, train_loader, num_epochs, prev_params)
    prev_params = deepcopy(state.params)
    prev_hidden_means, prev_hidden_logvars, prev_last_means, prev_last_logvars = extract_means_and_logvars(prev_params)

    for i, coreset_loader in enumerate(coresets):
        state = create_train_state(model, prev_params, learning_rate=1e-3)
        key, subkey = random.split(key)
        state = train_Dt(subkey, state, i, coreset_loader, num_epochs, prev_params)

        test_loader = PermutedLoader(test_data, num_samples=num_pred_samples, permutation=permutations[i], batch_size=128, shuffle=False)
        key, subkey = random.split(key)
        accuracy = eval_Dt(subkey, state, i, test_loader)
        print(f"Task {i} accuracy: {accuracy}")
