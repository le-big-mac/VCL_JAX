import jax.numpy as jnp
import optax


def kl_divergence(m, v, m0, v0):
    # Compute the KL divergence between two normal distributions
    if len(m.shape) == 1:
        const_term = -0.5 * m.shape[0]  # For bias
    else:
        const_term = -0.5 * m.shape[0] * m.shape[1]  # For kernel

    log_std_diff = 0.5 * jnp.sum(v0 - v)
    mu_diff_term = 0.5 * jnp.sum((jnp.exp(v) + (m - m0)**2) / jnp.exp(v0))
    kl = const_term + log_std_diff + mu_diff_term
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


def accuracy(logits, targets):
    return jnp.mean(jnp.argmax(logits, axis=-1) == targets)
