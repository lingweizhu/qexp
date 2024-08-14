import numpy as np
import torch
import torch.nn as nn
from core.network import network_utils, network_bodies

class Beta(nn.Module):
    def __init__(self, device, observation_dim, action_dim, arch,
                 action_min=-1, action_max=1, init_type='xavier'):
        super(Beta, self).__init__()

        self.action_dim = action_dim

        self.base_network = network_bodies.FCBody(device, observation_dim, hidden_units=tuple(arch), init_type=init_type)
        self.alpha_head = network_utils.layer_init_xavier(nn.Linear(arch[-1], action_dim))
        self.beta_head = network_utils.layer_init_xavier(nn.Linear(arch[-1], action_dim))

        self.head_activation_fn = torch.nn.functional.softplus
        self.beta_param_bias = torch.tensor(1.)
        self.to(device)
        self.device = device
        self.epsilon = 1e-8
        self.float32_eps = 10 * np.finfo(np.float32).eps

        self.action_scale = action_max - action_min
        self.action_bias = - (action_max - action_min) / 2.

    def forward(self, observation):
        base = self.base_network(observation)
        alpha = self.head_activation_fn(self.alpha_head(base)) + self.epsilon
        beta = self.head_activation_fn(self.beta_head(base)) + self.epsilon
        alpha += self.beta_param_bias
        beta += self.beta_param_bias

        dist = torch.distributions.Beta(alpha, beta)
        dist = torch.distributions.Independent(dist, 1)
        return dist

    def rsample(self, observation):
        dist = self.forward(observation)
        out = dist.rsample()  # samples of alpha and beta
        logp = dist.log_prob(torch.clamp(out, 0 + self.float32_eps, 1 - self.float32_eps))
        logp = logp.view((logp.shape[0], 1))
        action = out * self.action_scale + self.action_bias
        return action, logp

    def sample(self, observation, deterministic=False):
        dist = self.forward(observation)
        out = dist.sample()  # samples of alpha and beta
        with torch.no_grad():
            logp = dist.log_prob(torch.clamp(out, 0 + self.float32_eps, 1 - self.float32_eps))
        logp = logp.view((logp.shape[0], 1))
        action = out * self.action_scale + self.action_bias
        return action, logp

    def log_prob(self, observation, action):
        out = (action - self.action_bias) / self.action_scale
        out = torch.clamp(out, 0, 1)
        dist = self.forward(observation)
        logp = dist.log_prob(torch.clamp(out, 0 + self.float32_eps, 1 - self.float32_eps))
        logp = logp.view(-1,1)
        return logp