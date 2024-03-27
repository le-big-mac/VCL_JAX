import jax.numpy as jnp
from flax import linen as nn
from flax.training import common_utils


def kl_divergence(m, v, m0, v0):
    # Compute the KL divergence between two normal distributions
    kl = 0.5 * (jnp.sum(v0 - v - 1 + (jnp.exp(v) + (m0 - m)**2) / jnp.exp(v0)))
    return kl


def total_kl_divergence(new_params, prev_params):
    kl = 0.0
    for key in prev_params.keys():
        kl += kl_divergence(new_params[key]['kernel_mean'], new_params[key]['kernel_logvar'], prev_params[key]['kernel_mean'], prev_params[key]['kernel_logvar'])
        kl += kl_divergence(new_params[key]['bias_mean'], new_params[key]['bias_logvar'], prev_params[key]['bias_mean'], prev_params[key]['bias_logvar'])

    diff = set(new_params.keys()) - set(prev_params.keys())
    for key in diff:
        kl += kl_divergence(new_params[key]['kernel_mean'], new_params[key]['kernel_logvar'], 0.0, 0.0)
        kl += kl_divergence(new_params[key]['bias_mean'], new_params[key]['bias_logvar'], 0.0, 0.0)

    return kl


def loglikelihood(logits, targets, num_samples, num_classes=10):
    targets = common_utils.onehot(targets, num_classes)
    targets = jnp.repeat(targets[jnp.newaxis, ...], num_samples, axis=0)
    log_lik = -nn.softmax_cross_entropy_with_logits(logits, targets).mean()
    return log_lik


def loss_fn(state, logits, targets, prev_params):
    kl = total_kl_divergence(state.params, prev_params)
    log_lik = loglikelihood(logits, targets, state.num_train_samples, state.output_size)
    return log_lik + kl / targets.shape[0]


def accuracy(logits, targets):
    return jnp.mean(jnp.argmax(logits, axis=-1) == targets)