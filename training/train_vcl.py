import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

from training.utils import total_kl_divergence, loglikelihood


def create_train_state(model, params, learning_rate):
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step_mfvi(rng, state, task_idx, data, targets, prev_params, train_set_size):
    def get_loss(params):
        logits = state.apply_fn({"params": params}, data, task_idx, training=True, rngs={"samples": rng})
        kl = total_kl_divergence(params, prev_params) / train_set_size
        loglik = -loglikelihood(logits, targets)
        return kl + loglik
    grad_fn = jax.value_and_grad(get_loss)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def train_step(state, data, targets):
    def get_loss(params):
        logits = state.apply_fn({"params": params}, data)
        return -loglikelihood(logits, targets)
    grad_fn = jax.value_and_grad(get_loss)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(rng, state, task_idx, data):
    logits = state.apply_fn({"params": state.params}, data, task_idx, training=False, rngs={"samples": rng})
    return jnp.mean(logits, axis=0)


def train_Dt(rng, state, task_idx, head_idx, task_loader, num_epochs, prev_params, train_set_size):
    for i in range(num_epochs):
        epoch_loss = 0
        for data, targets in task_loader:
            rng, subkey = jax.random.split(rng)
            state, loss = train_step_mfvi(subkey, state, head_idx, data, targets, prev_params, train_set_size)
            epoch_loss += loss

        print(f"Task {task_idx}: Epoch {i+1}, Loss: {epoch_loss / len(task_loader)}")

    return state


def train_standard(state, loader, num_epochs):
    for i in range(num_epochs):
        epoch_loss = 0
        for data, targets in loader:
            state, loss = train_step(state, data, targets)
            epoch_loss += loss

        print(f"Epoch {i+1}, Loss: {epoch_loss / len(loader)}")

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
