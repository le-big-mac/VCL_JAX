import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from tqdm import tqdm

from training.utils import loss_fn


def create_train_state(model, params, learning_rate):
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(rng, state, task_idx, data, targets, prev_params):
    def get_loss(params):
        logits = state.apply_fn({"params": params}, data, task_idx, training=True, rngs={"samples": rng})
        return loss_fn(state, logits, targets, prev_params)
    grad_fn = jax.value_and_grad(get_loss)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def eval_step(rng, state, task_idx, data):
    logits = state.apply_fn({"params": state.params}, data, task_idx, training=False, rngs={"samples": rng})
    return jnp.mean(logits, axis=0)


def train_Dt(rng, state, task_idx, task_loader, num_epochs, prev_params):
    for _ in tqdm(range(num_epochs)):
        for data, targets in task_loader:
            rng, subkey = jax.random.split(rng)
            state = train_step(subkey, state, task_idx, data, targets, prev_params)

    return state


def eval_Dt(rng, state, task_idx, task_loader):
    total = 0
    correct = 0
    for data, targets in task_loader:
        targets = targets.squeeze()
        rng, subkey = jax.random.split(rng)
        logits = eval_step(subkey, state, task_idx, data)
        total += len(targets)
        correct += (logits.argmax(axis=-1) == targets).sum()

    return correct / total