import jax
import jax.numpy as jnp
from flax import linen as nn

from .layers import MFVI_Dense


class MFVI_NN(nn.Module):
    hidden_size: list[int]
    output_size: int
    previous_mean_hidden: tuple[list[jax.Array], list[jax.Array]]
    previous_logvar_hidden: tuple[list[jax.Array], list[jax.Array]]
    previous_mean_last: tuple[list[jax.Array], list[jax.Array]]
    previous_logvar_last: tuple[list[jax.Array], list[jax.Array]]
    num_train_samples: int = 10
    num_pred_samples: int = 100
    prior_mean : int = 0.0
    prior_logvar : int = 0.0

    def setup(self):
        self.hidden_layers = [MFVI_Dense(self.hidden_size[i], self.previous_mean_hidden[0][i], self.previous_logvar_hidden[0][i], self.previous_mean_hidden[1][i], self.previous_logvar_hidden[1][i]) for i in range(len(self.hidden_size))]

        task_heads = [MFVI_Dense(self.output_size, W_m_l, W_v_l, b_m_l, b_v_l) for W_m_l, W_v_l, b_m_l, b_v_l in zip(self.previous_mean_last[0], self.previous_logvar_last[0], self.previous_mean_last[1], self.previous_logvar_last[1])]

        W_m_l = jnp.full((self.hidden_size[-1], self.output_size), self.prior_mean)
        W_v_l = jnp.full((self.hidden_size[-1], self.output_size), self.prior_logvar)
        b_m_l = jnp.full((self.output_size,), self.prior_mean)
        b_v_l = jnp.full((self.output_size,), self.prior_logvar)
        new_head = MFVI_Dense(self.output_size, W_m_l, W_v_l, b_m_l, b_v_l)
        self.task_heads = task_heads + [new_head]

    def __call__(self, inputs, task_idx, training=False):
        num_samples = self.num_train_samples if training else self.num_pred_samples

        rng = self.make_rng('samples')
        sample_keys = jax.random.split(rng, num_samples)
        sample_fn = jax.vmap(self._single_sample, in_axes=(None, None, 0))

        return sample_fn(inputs, task_idx, sample_keys)

    def _single_sample(self, inputs, task_idx, sample_key):
        def head_fn(i):
            return lambda mdl, x, k: mdl.task_heads[i](x, k)
        branches = [head_fn(i) for i in range(len(self.task_heads))]

        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, sample_key)
            x = nn.relu(x)

        # run all branches on init
        if self.is_mutable_collection('params'):
            for branch in branches:
                _ = branch(self, x, sample_key)

        return nn.switch(task_idx, branches, self, x, sample_key)





# class MFVI_NN(nn.Module):
#     input_size: int
#     hidden_size: list[int]
#     output_size: int
#     no_train_samples: int = 10
#     no_pred_samples: int = 100
#     previous_mean_hidden: tuple[list[jax.Array], list[jax.Array]] = None
#     previous_var_hidden: tuple[list[jax.Array], list[jax.Array]] = None
#     previous_means_last: tuple[list[jax.Array], list[jax.Array]] = None
#     previous_vars_last: tuple[list[jax.Array], list[jax.Array]] = None
#     prior_mean : int = 0.0
#     prior_var : int = 1.0

#     def setup(self):
#         layer_sizes = [self.input_size] + self.hidden_size
#         num_hidden = len(layer_sizes)

#         if self.previous_mean_hidden is None:
#             self.W_hidden = [self.param(f'W_m_{i}', nn.initializers.normal(stddev=0.1), (layer_sizes[i], layer_sizes[i+1]))
#                         for i in range(num_hidden - 1)]
#             self.b_hidden = [self.param(f'b_m_{i}', nn.initializers.normal(stddev=0.1), (layer_sizes[i+1],))
#                         for i in range(num_hidden - 1)]
#             self.W_hidden_var = [self.param(f'W_v_{i}', lambda _, shape: jnp.full(shape, -6.0), (layer_sizes[i], layer_sizes[i+1])) for i in range(num_hidden - 1)]
#             self.b_hidden_var = [self.param(f'b_v_{i}', lambda _, shape: jnp.full(shape, -6.0), (layer_sizes[i+1],)) for i in range(num_hidden - 1)]
#         else:
#             self.W_hidden = [self.param(f'W_m_{i}', lambda _: self.previous_mean_hidden[0][i]) for i in range(num_hidden - 1)]
#             self.b_hidden = [self.param(f'b_m_{i}', lambda _: self.previous_mean_hidden[1][i]) for i in range(num_hidden - 1)]
#             self.W_hidden_var = [self.param(f'W_v_{i}', lambda _: jnp.exp(self.previous_var_hidden[0][i])) for i in range(num_hidden - 1)]
#             self.b_hidden_var = [self.param(f'b_v_{i}', lambda _: jnp.exp(self.previous_var_hidden[1][i])) for i in range(num_hidden - 1)]

#         self.W_m = [self.param(f'W_m_{i}', nn.initializers.normal(stddev=0.1), (din, dout))
#                     for i, (din, dout) in enumerate(zip(self.hidden_size[:-1], self.hidden_size[1:]))]
#         self.b_m = [self.param(f'b_m_{i}', nn.initializers.normal(stddev=0.1), (dout,))
#                     for i, dout in enumerate(self.hidden_size[1:])]
#         self.W_last_m = self.param('W_last_m', nn.initializers.normal(stddev=0.1), (self.hidden_size[-1], self.output_size))
#         self.b_last_m = self.param('b_last_m', nn.initializers.normal(stddev=0.1), (self.output_size,))

#         self.W_v = [self.param(f'W_v_{i}', lambda _, shape: jnp.full(shape, -6.0), (din, dout))
#                     for i, (din, dout) in enumerate(zip(self.hidden_size[:-1], self.hidden_size[1:]))]
#         self.b_v = [self.param(f'b_v_{i}', lambda _, shape: jnp.full(shape, -6.0), (dout,))
#                     for i, dout in enumerate(self.hidden_size[1:])]
#         self.W_last_v = self.param('W_last_v', lambda _, shape: jnp.full(shape, -6.0), (self.hidden_size[-1], self.output_size))
#         self.b_last_v = self.param('b_last_v', lambda _, shape: jnp.full(shape, -6.0), (self.output_size,))

#     def __call__(self, inputs, rng_key):
#         return self._predict(inputs, rng_key, self.no_pred_samples)

#     def _predict(self, inputs, rng_key, no_samples):
#         def apply_layer(layer_m, layer_v, activations, rng_key):
#             rng_key, w_key, b_key = random.split(rng_key, 3)
#             weights = layer_m + jnp.exp(0.5 * layer_v) * random.normal(w_key, layer_m.shape)
#             biases = random.normal(b_key, layer_v.shape)
#             return jnp.dot(activations, weights) + biases, rng_key

#         activations = jnp.repeat(inputs[jnp.newaxis, ...], no_samples, axis=0)
#         rng_keys = random.split(rng_key, len(self.W_m) + 1)

#         for i, (W_m, W_v, b_m, b_v, rng_key) in enumerate(zip(self.W_m, self.W_v, self.b_m, self.b_v, rng_keys[:-1])):
#             activations, rng_key = apply_layer(W_m, W_v, activations, rng_key)
#             activations += b_m + jnp.exp(0.5 * b_v) * random.normal(rng_key, b_m.shape)
#             activations = nn.relu(activations)

#         rng_key = rng_keys[-1]
#         output, _ = apply_layer(self.W_last_m, self.W_last_v, activations, rng_key)
#         output += self.b_last_m + jnp.exp(0.5 * self.b_last_v) * random.normal(rng_key, self.b_last_m.shape)
#         return output

#     def _log_likelihood(self, inputs, targets, rng_key):
#         preds = self._predict(inputs, rng_key, self.no_train_samples)
#         targets = jnp.repeat(targets[jnp.newaxis, ...], self.no_train_samples, axis=0)
#         log_lik = -nn.softmax_cross_entropy_with_logits(preds, targets).mean()
#         return log_lik

#     def _kl_divergence(self, m, v, m_prior, v_prior):
#         return 0.5 * jnp.sum(v_prior - v - 1.0 + jnp.exp(v) / v_prior + (m - m_prior)**2 / v_prior)

#     def kl_term(self, prior_mean, prior_var):
#         kl = 0.0
#         for W_m, W_v, b_m, b_v in zip(self.W_m, self.W_v, self.b_m, self.b_v):
#             kl += self._kl_divergence(W_m, W_v, prior_mean, prior_var)
#             kl += self._kl_divergence(b_m, b_v, prior_mean, prior_var)
#         kl += self._kl_divergence(self.W_last_m, self.W_last_v, prior_mean, prior_var)
#         kl += self._kl_divergence(self.b_last_m, self.b_last_v, prior_mean, prior_var)
#         return kl

#     def loss_fn(self, params, inputs, targets, prior_mean, prior_var, rng_key):
#         log_lik = self._log_likelihood(inputs, targets, rng_key)
#         kl_term = self.kl_term(prior_mean, prior_var)
#         return -log_lik + 1.0 / inputs.shape[0] * kl_term

#     def train_step(self, optimizer, batch, prior_mean, prior_var, rng_key):
#         inputs, targets = batch
#         loss_fn_partial = partial(self.loss_fn, inputs=inputs, targets=targets, prior_mean=prior_mean, prior_var=prior_var)
#         loss, grads = jax.value_and_grad(loss_fn_partial)(optimizer.target, rng_key)
#         optimizer = optimizer.apply_gradient(grads)
#         return optimizer, loss

#     @partial(jax.jit, static_argnums=(0,))
#     def predict(self, inputs, params, rng_key):
#         return self.apply({'params': params}, inputs, rng_key, rngs={'dropout': rng_key})
