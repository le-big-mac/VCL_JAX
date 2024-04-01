import jax
from flax import linen as nn

from .layers import MFVI_Dense


class Standard_NN(nn.Module):
    hidden_size: list[int]
    output_size: int

    def setup(self):
        self.hidden_layers = [
            nn.Dense(self.hidden_size[i])
            for i in range(len(self.hidden_size))
            ]

        self.final = nn.Dense(self.output_size)

    def __call__(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
            x = nn.relu(x)

        return self.final(x)


def extract_means(params):
    hidden_means = ([], [])
    last_means = ([], [])

    i = 0
    while True:
        try:
            hidden_means[0].append(params[f'hidden_layers_{i}']['kernel'])
            hidden_means[1].append(params[f'hidden_layers_{i}']['bias'])
            i += 1
        except KeyError:
            break

    last_means[0].append(params['final']['kernel'])
    last_means[1].append(params['final']['bias'])

    return hidden_means, last_means


class MFVI_NN(nn.Module):
    hidden_size: list[int]
    output_size: int
    previous_mean_hidden: tuple[list[jax.Array], list[jax.Array]]
    previous_logvar_hidden: tuple[list[jax.Array], list[jax.Array]]
    previous_mean_last: tuple[list[jax.Array], list[jax.Array]]
    previous_logvar_last: tuple[list[jax.Array], list[jax.Array]]
    num_train_samples: int = 10
    num_pred_samples: int = 100

    def setup(self):
        self.hidden_layers = [
            MFVI_Dense(self.hidden_size[i],
                       self.previous_mean_hidden[0][i],
                       self.previous_logvar_hidden[0][i],
                       self.previous_mean_hidden[1][i],
                       self.previous_logvar_hidden[1][i])
            for i in range(len(self.hidden_size))
            ]

        self.task_heads = [
            MFVI_Dense(self.output_size, W_m_l, W_v_l, b_m_l, b_v_l)
            for W_m_l, W_v_l, b_m_l, b_v_l in
            zip(self.previous_mean_last[0],
                self.previous_logvar_last[0],
                self.previous_mean_last[1],
                self.previous_logvar_last[1])
            ]

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


def extract_means_and_logvars(params):
    hidden_means = ([], [])
    hidden_logvars = ([], [])
    last_means = ([], [])
    last_logvars = ([], [])

    i = 0
    while True:
        try:
            hidden_means[0].append(params[f'hidden_layers_{i}']['kernel_mean'])
            hidden_means[1].append(params[f'hidden_layers_{i}']['bias_mean'])
            hidden_logvars[0].append(params[f'hidden_layers_{i}']['kernel_logvar'])
            hidden_logvars[1].append(params[f'hidden_layers_{i}']['bias_logvar'])
            i += 1
        except KeyError:
            break

    i = 0
    while True:
        try:
            last_means[0].append(params[f'task_heads_{i}']['kernel_mean'])
            last_means[1].append(params[f'task_heads_{i}']['bias_mean'])
            last_logvars[0].append(params[f'task_heads_{i}']['kernel_logvar'])
            last_logvars[1].append(params[f'task_heads_{i}']['bias_logvar'])
            i += 1
        except KeyError:
            break

    return hidden_means, hidden_logvars, last_means, last_logvars

