import jax
import jax.numpy as jnp
import flax.linen as nn


class MFVI_Dense(nn.Module):
    features: int
    kernel_mean_init: jnp.ndarray
    kernel_logvar_init: jnp.ndarray
    bias_mean_init: jnp.ndarray
    bias_logvar_init: jnp.ndarray

    @nn.compact
    def __call__(self, inputs, rng=None):
        kernel_mean = self.param('kernel_mean', lambda *_: self.kernel_mean_init)
        kernel_logvar = self.param('kernel_logvar', lambda *_: self.kernel_logvar_init, self.kernel_logvar_init.shape)
        kernel_std = jnp.exp(0.5 * kernel_logvar)

        bias_mean = self.param('bias_mean', lambda *_: self.bias_mean_init, self.bias_mean_init.shape)
        bias_logvar = self.param('bias_logvar', lambda *_: self.bias_logvar_init, self.bias_logvar_init.shape)
        bias_std = jnp.exp(0.5 * bias_logvar)

        kernel_samples = jax.random.normal(rng, kernel_mean.shape) * kernel_std + kernel_mean
        bias_samples = jax.random.normal(rng, bias_mean.shape) * bias_std + bias_mean

        dense = nn.Dense(features=self.features)
        return dense.apply({'params': {'kernel': kernel_samples, 'bias': bias_samples}}, inputs)