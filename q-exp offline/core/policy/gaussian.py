import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical

from core.network import network_utils, network_bodies
from core.utils import torch_utils


class SquashedGaussian(nn.Module):
    def __init__(self, device, observation_dim, action_dim, arch, action_min, action_max):
        super(SquashedGaussian, self).__init__()
        assert action_min == -1 and action_max == 1 # TODO: accept any action range
        if len(arch) > 0:
            self.base_network = network_bodies.FCBody(device, observation_dim, hidden_units=tuple(arch), init_type='xavier')
            self.mean_head = network_utils.layer_init_xavier(nn.Linear(arch[-1], action_dim))
            self.logstd_head = network_utils.layer_init_xavier(nn.Linear(arch[-1], action_dim))
        else:
            raise NotImplementedError
        self.to(device)
        self.epsilon = 1e-6

    def forward(self, observation):
        base = self.base_network(observation)
        mean = self.mean_head(base)
        log_std = torch.clamp(self.logstd_head(base), min=-20, max=2)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        normal = torch.distributions.Independent(normal, 1)
        return normal

    def rsample(self, observation):
        normal = self.forward(observation)
        out = normal.rsample()
        tanhout = torch.tanh(out)
        logp = normal.log_prob(out)
        logp -= torch.log((1 - tanhout.pow(2)) + self.epsilon).sum(axis=-1)
        logp = logp.view((logp.shape[0], 1))
        return tanhout, logp

    def sample(self, observation, deterministic=False):
        normal = self.forward(observation)
        out = normal.sample()
        tanhout = torch.tanh(out)
        with torch.no_grad():
            logp = normal.log_prob(out)
        logp -= torch.log((1 - tanhout.pow(2)) + self.epsilon).sum(axis=-1)
        logp = logp.view((logp.shape[0], 1))
        return tanhout, logp

    def log_prob(self, observation, action):
        normal = self.forward(observation)
        tanhout = action
        out = torch.atanh(torch.clamp(tanhout, -1.0 + self.epsilon, 1.0 - self.epsilon))
        logp = normal.log_prob(out)
        logp -= torch.log((1 - tanhout.pow(2)) + self.epsilon).sum(axis=-1).reshape(logp.shape)
        logp = logp.view(-1, 1)
        return logp

    def distribution(self, x, dim=-1):
        with torch.no_grad():
            base = self.base_network(x)
            mean = self.mean_head(base)
            log_std = torch.clamp(self.logstd_head(base), min=-20, max=2)
            std = log_std.exp()
        if dim == -1:
            normal = torch.distributions.Normal(mean, std)
            dist = torch.distributions.Independent(normal, 1)
            return dist, mean, std, None
        else:
            mean = mean[:, dim: dim + 1]
            std = std[:, dim: dim + 1]
            dist = torch.distributions.Normal(mean, std)
            return dist, mean, std, None


class Gaussian(SquashedGaussian):
    def __init__(self, device, observation_dim, action_dim, arch, action_min, action_max):
        super(Gaussian, self).__init__(device, observation_dim, action_dim, arch, action_min, action_max)


    def rsample(self, observation):
        normal = self.forward(observation)
        out = normal.rsample()
        logp = normal.log_prob(out)
        logp = logp.view((logp.shape[0], 1))
        return out, logp

    def sample(self, observation, deterministic=False):
        normal = self.forward(observation)
        out = normal.sample()
        with torch.no_grad():
            logp = normal.log_prob(out)
        logp = logp.view((logp.shape[0], 1))
        return out, logp

    def log_prob(self, observation, action):
        normal = self.forward(observation)
        out = action
        logp = normal.log_prob(out)
        logp = logp.view(-1, 1)
        return logp


class MLPCont(nn.Module):
    def __init__(self, device, obs_dim, act_dim, hidden_sizes, action_range=1.0, init_type='xavier'):
        super().__init__()
        self.device = device
        body = network_bodies.FCBody(device, obs_dim, hidden_units=tuple(hidden_sizes), init_type=init_type)
        body_out = obs_dim if hidden_sizes==[] else hidden_sizes[-1]
        self.body = body
        self.mu_layer = network_utils.layer_init_xavier(nn.Linear(body_out, act_dim))
        self.log_std_logits = nn.Parameter(torch.zeros(act_dim, requires_grad=True))
        self.min_log_std = -6
        self.max_log_std = 0
        self.action_range = action_range # TODO: accept any action range

    """https://github.com/hari-sikchi/AWAC/blob/3ad931ec73101798ffe82c62b19313a8607e4f1e/core.py#L91"""
    def forward(self, obs, deterministic=False):
        if not isinstance(obs, torch.Tensor): obs = torch_utils.tensor(obs, self.device)
        net_out = self.body(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.action_range

        log_std = torch.sigmoid(self.log_std_logits)
        log_std = self.min_log_std + log_std * (self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        return pi_distribution, mu

    def rsample(self, obs):
        dist, _ = self.forward(obs)
        pi_action = dist.rsample()
        logp_pi = dist.log_prob(pi_action).sum(axis=-1, keepdim=True)
        return pi_action, logp_pi

    def sample(self, obs, deterministic=False):
        dist, mu = self.forward(obs, deterministic=deterministic)
        if deterministic:
            pi_action = mu
        else:
            pi_action = dist.sample()
        with torch.no_grad():
            logp_pi = dist.log_prob(pi_action).sum(axis=-1, keepdim=True)
        return pi_action, logp_pi

    def log_prob(self, obs, actions):
        if not isinstance(obs, torch.Tensor): obs = torch_utils.tensor(obs, self.device)
        if not isinstance(actions, torch.Tensor): actions = torch_utils.tensor(actions, self.device)
        net_out = self.body(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.action_range
        log_std = torch.sigmoid(self.log_std_logits)
        # log_std = self.log_std_layer(net_out)
        log_std = self.min_log_std + log_std * (
            self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(actions).sum(axis=-1, keepdim=True)
        return logp_pi


class MLPDiscrete(nn.Module):
    def __init__(self, device, obs_dim, act_dim, hidden_sizes, init_type='xavier'):
        super().__init__()
        self.device = device
        body = network_bodies.FCBody(device, obs_dim, hidden_units=tuple(hidden_sizes), init_type=init_type)
        body_out = obs_dim if hidden_sizes==[] else hidden_sizes[-1]
        self.body = body
        self.mu_layer = network_utils.layer_init_xavier(nn.Linear(body_out, act_dim))
        self.log_std_logits = nn.Parameter(torch.zeros(act_dim, requires_grad=True))
        self.min_log_std = -6
        self.max_log_std = 0
    
    def forward(self, obs, deterministic=True):
        net_out = self.body(obs)
        probs = self.mu_layer(net_out)
        probs = F.softmax(probs, dim=1)
        m = Categorical(probs)
        action = m.sample()
        logp = m.log_prob(action)
        return action, logp

    def rsample(self, obs):
        return self.forward(obs)
    
    def log_prob(self, obs, actions):
        if not isinstance(obs, torch.Tensor): obs = torch_utils.tensor(obs, self.device)
        if not isinstance(actions, torch.Tensor): actions = torch_utils.tensor(actions, self.device)
        net_out = self.body(obs)
        probs = self.mu_layer(net_out)
        probs = F.softmax(probs, dim=1)
        m = Categorical(probs)
        logp_pi = m.log_prob(actions)
        return logp_pi
    