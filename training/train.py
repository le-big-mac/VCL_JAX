import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

from utils import loss_fn


def create_train_state(rng, learning_rate, model):
    params = model.init(rng, jnp.ones([1, 1]))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, task_idx, batch, prev_params):
    data, labels = batch
    logits = state.apply_fn({"params": state.params}, data, task_idx, training=True)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state, logits, labels, prev_params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def eval_step(state, task_idx, batch):
    data, labels = batch
    logits = state.apply_fn({"params": state.params}, data, task_idx, training=False)
    return logits


def train_Dt(state, task_idx, task_loader, num_epochs, prev_params):
    for _ in range(num_epochs):
        for batch in task_loader:
            state = train_step(state, task_idx, batch, prev_params)

    return state


def train_coreset(state, coreset, num_epochs, prev_params):
    for _ in range(num_epochs):
        for task_idx, coreset_data in enumerate(coreset):
            for batch in coreset_data:
                state = train_step(state, task_idx, batch, prev_params)

    return state
