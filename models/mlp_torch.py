import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class MFVI_NN(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_sizes: list[int],
                 output_size: int,
                 num_train_samples: int = 10,
                 num_pred_samples: int = 100,
                 prev_hidden_means: tuple[list[torch.tensor], list[torch.tensor]] = None,
                 prev_hidden_log_vars: tuple[list[torch.tensor], list[torch.tensor]] = None,
                 prev_last_means: tuple[list[torch.tensor], list[torch.tensor]] = None,
                 prev_last_log_vars: tuple[list[torch.tensor], list[torch.tensor]] = None,
                 prior_mean: int = 0,
                 prior_var:int = 1):
        super(MFVI_NN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_train_samples = num_train_samples
        self.num_pred_samples = num_pred_samples

        # Create the weights and biases of the network
        self.W_m, self.b_m, self.W_v, self.b_v,  = self.create_hidden_weights(input_size, hidden_sizes, prev_hidden_means, prev_hidden_log_vars)
        self.W_last_m, self.b_last_m, self.W_last_v, self.b_last_v = self.create_last_weights(hidden_sizes[-1], output_size, prev_last_means, prev_last_log_vars)

    def create_hidden_weights(self, in_dim, hidden_sizes, prev_means, prev_log_vars):
        # Create the weights and biases for each layer
        layer_sizes = [in_dim] + hidden_sizes

        if prev_means is None:
            W_m = nn.ParameterList([nn.Parameter(torch.randn((din, dout)) * 0.1)
                                    for din, dout in zip(layer_sizes[:-1], layer_sizes[1:])])
            b_m = nn.ParameterList([nn.Parameter(torch.randn((dout,)) * 0.1) for dout in layer_sizes[1:]])
        else:
            W_m = nn.ParameterList([nn.Parameter(prev_means[0][i]) for i in range(len(hidden_sizes))])
            b_m = nn.ParameterList([nn.Parameter(prev_means[1][i]) for i in range(len(hidden_sizes))])

        if prev_log_vars is None:
            W_v = nn.ParameterList([nn.Parameter(torch.full((din, dout), -6.0))
                                    for din, dout in zip(layer_sizes[:-1], layer_sizes[1:])])
            b_v = nn.ParameterList([nn.Parameter(torch.full((dout,), -6.0)) for dout in layer_sizes[1:]])
        else:
            W_v = nn.ParameterList([nn.Parameter(torch.exp(prev_log_vars[0][i]))
                                    for i in range(len(hidden_sizes))])
            b_v = nn.ParameterList([nn.Parameter(torch.exp(prev_log_vars[1][i]))
                                    for i in range(len(hidden_sizes))])

        return W_m, b_m, W_v, b_v

    def create_last_weights(pen_dim, out_dim, prev_means, prev_log_variances):
        pass

    def forward(self, x, task_idx):
        # Perform prediction using the mean-field approximation
        num_samples = self.num_train_samples if self.training else self.num_pred_samples
        pred = self._prediction(x, task_idx, num_samples)
        return pred

    def _prediction(self, inputs, task_idx, no_samples):
        # Sample the weights and biases and compute the network output
        activations = inputs.unsqueeze(0).expand(no_samples, -1, -1)
        for i in range(self.no_layers - 1):
            # Sample the weights and biases for the current layer
            weights = self.W_m[i] + torch.randn_like(self.W_m[i]) * torch.exp(0.5 * self.W_v[i])
            biases = self.b_m[i] + torch.randn_like(self.b_m[i]) * torch.exp(0.5 * self.b_v[i])
            # Compute the pre-activations and apply the activation function
            pre_activations = torch.einsum('mni,mio->mno', activations, weights) + biases
            activations = F.relu(pre_activations)

        # Sample the weights and biases for the last layer
        weights_last = self.W_last_m[task_idx] + torch.randn_like(self.W_last_m[task_idx]) * torch.exp(0.5 * self.W_last_v[task_idx])
        biases_last = self.b_last_m[task_idx] + torch.randn_like(self.b_last_m[task_idx]) * torch.exp(0.5 * self.b_last_v[task_idx])
        # Compute the output of the last layer
        output = torch.einsum('mni,mio->mno', activations, weights_last) + biases_last
        return output





def _logpred(self, inputs, targets, task_idx):
    # Compute the log-likelihood of the predictions
    pred = self._prediction(inputs, task_idx, self.no_train_samples)
    targets = targets.unsqueeze(0).expand(self.no_train_samples, -1, -1)
    log_lik = -F.cross_entropy(pred, targets, reduction='mean')
    return log_lik

def _KL_term(self):
    # Compute the KL divergence between the variational distributions and the priors
    kl = 0
    for i in range(self.no_layers - 1):
        kl += self._kl_divergence(self.W_m[i], self.W_v[i], self.prior_W_m[i], self.prior_W_v[i])
        kl += self._kl_divergence(self.b_m[i], self.b_v[i], self.prior_b_m[i], self.prior_b_v[i])

    for i in range(len(self.W_last_m)):
        kl += self._kl_divergence(self.W_last_m[i], self.W_last_v[i], self.prior_W_last_m[i], self.prior_W_last_v[i])
        kl += self._kl_divergence(self.b_last_m[i], self.b_last_v[i], self.prior_b_last_m[i], self.prior_b_last_v[i])

    return kl

def _kl_divergence(self, m, v, m0, v0):
    # Compute the KL divergence between two normal distributions
    kl = 0.5 * (torch.sum(v0 - v - 1 + (v.exp() + (m0 - m)**2) / v0))
    return kl

def create_prior(self, in_dim, hidden_size, out_dim, prev_means, prev_log_variances, prior_mean, prior_var, is_variance=False):
    # Create the prior distributions for the weights and biases
    hidden_size = deepcopy(hidden_size)
    hidden_size.insert(0, in_dim)
    hidden_size.append(out_dim)

    W_m = []
    b_m = []
    W_last_m = []
    b_last_m = []

    for i in range(len(hidden_size) - 2):
        din = hidden_size[i]
        dout = hidden_size[i+1]
        if prev_means is not None and prev_log_variances is not None:
            Wi_m = prev_means[0][i]
            bi_m = prev_means[1][i]
            Wi_v = torch.exp(prev_log_variances[0][i])
            bi_v = torch.exp(prev_log_variances[1][i])
        else:
            Wi_m = torch.ones(din, dout) * prior_mean
            bi_m = torch.ones(dout) * prior_mean
            Wi_v = torch.ones(din, dout) * prior_var
            bi_v = torch.ones(dout) * prior_var

        W_m.append(Wi_m)
        b_m.append(bi_m)
        if is_variance:
            W_last_m.append(Wi_v)
            b_last_m.append(bi_v)

    if prev_means is not None and prev_log_variances is not None:
        for i in range(len(prev_means[2])):
            Wi_m = prev_means[2][i]
            bi_m = prev_means[3][i]
            Wi_v = torch.exp(prev_log_variances[2][i])
            bi_v = torch.exp(prev_log_variances[3][i])

            W_last_m.append(Wi_m)
            b_last_m.append(bi_m)
            if is_variance:
                W_last_m.append(Wi_v)
                b_last_m.append(bi_v)

    din = hidden_size[-2]
    dout = hidden_size[-1]
    Wi_m = torch.ones(din, dout) * prior_mean
    bi_m = torch.ones(dout) * prior_mean
    Wi_v = torch.ones(din, dout) * prior_var
    bi_v = torch.ones(dout) * prior_var

    W_last_m.append(Wi_m)
    b_last_m.append(bi_m)
    if is_variance:
        W_last_m.append(Wi_v)
        b_last_m.append(bi_v)

    return W_m, b_m, W_last_m, b_last_m