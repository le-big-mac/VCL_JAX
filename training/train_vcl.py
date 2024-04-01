import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
import optax

from training.utils import loss_fn, total_kl_divergence, loglikelihood


def create_train_state(model, params, learning_rate):
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step_old(rng, state, task_idx, data, targets, prev_params):
    def get_loss(params):
        logits = state.apply_fn({"params": params}, data, task_idx, training=True, rngs={"samples": rng})
        return loss_fn(state, logits, targets, prev_params)
    grad_fn = jax.value_and_grad(get_loss)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def train_step_vcl(rng, state, task_idx, data, targets, prev_params):
    def kl_loss(params):
        return total_kl_divergence(params, prev_params) / targets.shape[1]

    def log_lik_loss(params):
        logits = state.apply_fn({"params": params}, data, task_idx, training=True, rngs={"samples": rng})
        return -loglikelihood(logits, targets)

    kl, kl_grads = jax.value_and_grad(kl_loss)(state.params)
    log_lik, log_lik_grads = jax.value_and_grad(log_lik_loss)(state.params)

    state = state.apply_gradients(grads=kl_grads)
    state = state.apply_gradients(grads=log_lik_grads)

    return state, kl + log_lik


@jax.jit
def eval_step(rng, state, task_idx, data):
    logits = state.apply_fn({"params": state.params}, data, task_idx, training=False, rngs={"samples": rng})
    return jnp.mean(logits, axis=0)


def train_epoch(rng, epoch, state, task_idx, task_loader, prev_params):
    batch_losses = []
    for data, targets in task_loader:
        rng, subkey = jax.random.split(rng)
        state, loss = train_step_vcl(subkey, state, task_idx, data, targets, prev_params)
        batch_losses.append(loss)

    batch_losses_np = jax.device_get(batch_losses)
    epoch_loss = np.mean(batch_losses_np)
    print(f"Epoch {epoch} loss: {epoch_loss}")

    return state


def train_Dt(rng, state, task_idx, task_loader, num_epochs, prev_params):
    for i in range(num_epochs):
        rng, subkey = jax.random.split(rng)
        state = train_epoch(subkey, i, state, task_idx, task_loader, prev_params)

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