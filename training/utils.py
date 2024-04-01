import jax.numpy as jnp
import optax


def kl_divergence(m, v, m0, v0):
    if len(m.shape) == 1:
        const = m.shape[0] # For bias
    else:
        const = m.shape[0] * m.shape[1] # For kernel

    # Compute the KL divergence between two normal distributions
    kl = 0.5 * (jnp.sum(v0 - v - const + (jnp.exp(v) + (m0 - m)**2) / jnp.exp(v0)))
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


def loglikelihood(logits, targets):
    log_lik = -optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return log_lik


def loss_fn(state, logits, targets, prev_params):
    kl = total_kl_divergence(state.params, prev_params)
    log_lik = loglikelihood(logits, targets)
    return kl / targets.shape[0] - log_lik


def accuracy(logits, targets):
    return jnp.mean(jnp.argmax(logits, axis=-1) == targets)