from copy import deepcopy

import jax
from jax import random
import jax.numpy as jnp

from data.mnist import SampleLoader, combine_datasets
from models.mlp import MFVI_NN, Standard_NN, extract_means_and_logvars, extract_means
from training.train_vcl import create_train_state, train_Dt, eval_Dt, train_standard

def vcl(key, model_hparams, task_train_data, task_test_data, coreset_size, coreset_selection_fn, multi_head=False):
    input_size = model_hparams['input_size']
    hidden_size = model_hparams['hidden_size']
    output_size = model_hparams['output_size']
    num_train_samples = model_hparams['num_train_samples']
    num_pred_samples = model_hparams['num_pred_samples']
    num_epochs = model_hparams['num_epochs']
    batch_size = model_hparams['batch_size']

    sizes = [input_size] + hidden_size
    truncated_normal = jax.nn.initializers.truncated_normal(stddev=0.1)
    prev_hidden_logvars = (
        [jnp.full((din, dout), -6.) for din, dout in zip(sizes[:-1], sizes[1:])],
        [jnp.full(dout, -6.) for dout in sizes[1:]]
        )
    prev_last_logvars = (
        [jnp.full((hidden_size[-1], output_size), -6.)],
        [jnp.full(output_size, -6.)]
        )

    first_task = task_train_data[0]
    first_loader = SampleLoader(first_task, num_samples=1, batch_size=batch_size, shuffle=True)

    # Initialize first model using standard training
    std_model = Standard_NN(hidden_size, output_size)
    dummy_input = jnp.ones([1, 784])
    key, params_key = random.split(key)
    params = std_model.init({'params': params_key}, dummy_input)["params"]
    std_state = create_train_state(std_model, params, learning_rate=1e-2)
    key, subkey = random.split(key)
    std_state = train_standard(std_state, first_loader, num_epochs)
    prev_hidden_means, prev_last_means = extract_means(std_state.params)

    # Get initial MVFI model parameters
    init_model = MFVI_NN(hidden_size, output_size, prev_hidden_means,
                         prev_hidden_logvars, prev_last_means,
                         prev_last_logvars,
                         num_train_samples=num_train_samples,
                         num_pred_samples=num_pred_samples)
    key, params_key = random.split(key)
    params = init_model.init({'params': params_key}, dummy_input, task_idx=0)["params"]
    prev_params = deepcopy(params)

    coresets = []
    for task_idx, task in enumerate(task_train_data):
        train_data = task
        if coreset_size > 0:
            train_data, coreset_data = coreset_selection_fn(train_data, coreset_size)
            coresets.append(coreset_data)

        train_loader = SampleLoader(train_data, num_samples=num_train_samples, batch_size=batch_size, shuffle=True)

        if multi_head:
            key, kernel_key, bias_key = random.split(key, 3)
            prev_last_means[0].append(truncated_normal(kernel_key, (hidden_size[-1], output_size), jnp.float32))
            prev_last_means[1].append(truncated_normal(bias_key, (output_size,), jnp.float32))
            prev_last_logvars[0].append(jnp.full((hidden_size[-1], output_size), -6.))
            prev_last_logvars[1].append(jnp.full(output_size, -6.))

        model = MFVI_NN(hidden_size, output_size, prev_hidden_means,
                        prev_hidden_logvars, prev_last_means,
                        prev_last_logvars,
                        num_train_samples=num_train_samples,
                        num_pred_samples=num_pred_samples)
        dummy_input = jnp.ones([1, 784])
        key, params_key = random.split(key)
        params = model.init({'params': params_key}, dummy_input, task_idx=0)["params"]
        state = create_train_state(model, params, learning_rate=1e-2)

        head_idx = task_idx if multi_head else 0
        key, subkey = random.split(key)
        state = train_Dt(subkey, state, head_idx, train_loader, num_epochs, prev_params)
        prev_params = deepcopy(state.params)
        prev_hidden_means, prev_hidden_logvars, prev_last_means, prev_last_logvars = extract_means_and_logvars(prev_params)

        if not multi_head and coreset_size > 0 and len(coresets) > 0:
            coreset = combine_datasets(coresets)
            coreset_loader = SampleLoader(coreset, num_samples=num_train_samples, batch_size=batch_size, shuffle=True)
            key, subkey = random.split(key)
            state = train_Dt(subkey, state, 0, coreset_loader, num_epochs, prev_params)

        for i in range(task_idx + 1):
            head_idx = 0
            if multi_head and coreset_size > 0 and len(coresets) > 0:
                head_idx = i
                state = create_train_state(model, prev_params, learning_rate=1e-2)
                coreset_loader = SampleLoader(coresets[head_idx], num_samples=num_train_samples, batch_size=batch_size, shuffle=True)
                key, subkey = random.split(key)
                state = train_Dt(subkey, state, head_idx, coreset_loader, num_epochs, prev_params)

            test_loader = SampleLoader(task_test_data[i], num_samples=1, batch_size=batch_size, shuffle=False)
            key, subkey = random.split(key)
            accuracy = eval_Dt(subkey, state, head_idx, test_loader)
            print(f"Task {i} accuracy: {accuracy}")