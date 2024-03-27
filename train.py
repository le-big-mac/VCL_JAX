import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

from utils import loss_fn, accuracy


def create_train_state(rng, learning_rate, model):
    params = model.init(rng, jnp.ones([1, 1]))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, task_idx, batch, prev_params):
    logits = state.apply_fn({"params": state.params}, batch["x"], task_idx, training=True)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state, logits, batch["y"], prev_params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def eval_step(state, task_idx, batch):
    logits = state.apply_fn({"params": state.params}, batch["x"], task_idx, training=False)
    return logits


def train_Dt(state, task_idx, train_data, num_epochs, prev_params):
    for _ in range(num_epochs):
        for batch in train_data:
            state = train_step(state, task_idx, batch, prev_params)

    return state


def train_coreset(state, coreset, num_epochs, prev_params):
    for _ in range(num_epochs):
        for task_idx, coreset_data in enumerate(coreset):
            for batch in coreset_data:
                state = train_step(state, task_idx, batch, prev_params)

    return state


def train_model(state_rng, model, task_dataset, task_idx, coreset, num_epochs, learning_rate, prev_params):
    state = create_train_state(state_rng, learning_rate, model)

    state = train_Dt(state, task_idx, task_dataset, num_epochs, prev_params)
    state = train_coreset(state, coreset, num_epochs, prev_params)

    return state


def evaluate_model(state, test_data):
    # TODO: fix this
    for task_idx, test_task in enumerate(test_data):
        for batch in test_task:
            logits = eval_step(state, task_idx, batch)
            print(f"Task {task_idx} accuracy: {accuracy(logits, batch['y'])}")