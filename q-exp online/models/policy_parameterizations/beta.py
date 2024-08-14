import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../../utils")
from utils.nn_utils import weights_init_

class Beta(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, activation, action_space, init=None):
        super(Beta, self).__init__()

        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_head = nn.Linear(hidden_dim, num_actions)
        self.beta_head = nn.Linear(hidden_dim, num_actions)

        self.action_max = torch.FloatTensor(action_space.high)
        self.action_min = torch.FloatTensor(action_space.low)

        self.apply(lambda module: weights_init_(module, init, activation))

        self.head_activation_fn = torch.nn.functional.softplus
        self.beta_param_bias = torch.tensor(1.)
        # self.to(device)
        # self.device = device
        self.epsilon = 1e-8
        self.float32_eps = 10 * np.finfo(np.float32).eps

        self.action_scale = self.action_max - self.action_min
        self.action_bias = - (self.action_max - self.action_min) / 2.


        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state):

        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))

        alpha = self.head_activation_fn(self.alpha_head(x)) + self.epsilon
        beta = self.head_activation_fn(self.beta_head(x)) + self.epsilon
        alpha += self.beta_param_bias
        beta += self.beta_param_bias

        dist = torch.distributions.Beta(alpha, beta)
        dist = torch.distributions.Independent(dist, 1)
        return dist

    def rsample(self, state, num_samples=1):
        dist = self.forward(state)
        action = dist.rsample((num_samples,)) 

        if num_samples == 1:
            action = action.squeeze(0)

        log_prob = dist.log_prob(torch.clamp(action, 0 + self.float32_eps, 1 - self.float32_eps))
        if self.num_actions == 1:
            log_prob.unsqueeze(-1) 

        # action = action * self.action_scale + self.action_bias
        action = action * (self.action_max - self.action_min) + self.action_min
        """
        beta's mean does not equal its mode
        return mode as greedy action
        """
        greedy_action = dist.mode * (self.action_max - self.action_min) + self.action_min
        return action, log_prob, greedy_action

    def sample(self, state, num_samples=1):
        dist = self.forward(state)
        action = dist.sample((num_samples, ))  # samples of alpha and beta
        if num_samples == 1:
            action = action.squeeze(0)

        with torch.no_grad():
            log_prob = dist.log_prob(torch.clamp(action, 0 + self.float32_eps, 1 - self.float32_eps))

        # why does this exist. It does nothing
        if self.num_actions == 1:
            log_prob.unsqueeze(-1)

        if log_prob.ndim == 1:
            log_prob = log_prob.unsqueeze(-1)

        # action = action * self.action_scale + self.action_bias
        action = action * (self.action_max - self.action_min) + self.action_min
        """
        beta's mean does not equal its mode
        return mode as greedy action
        """
        greedy_action = dist.mode * (self.action_max - self.action_min) + self.action_min

        return action, log_prob, greedy_action


    def log_prob(self, state, action):
        out = (action - self.action_bias) / self.action_scale
        out = torch.clamp(out, 0, 1)
        dist = self.forward(state)
        log_prob = dist.log_prob(torch.clamp(out, 0 + self.float32_eps, 1 - self.float32_eps))
        if self.num_actions == 1:
            log_prob.unsqueeze(-1)
        return log_prob
    
    def entropy(self, states):
        # """GreedyAC style entropy computation"""
        # _, log_probs, _ = self.rsample(states, num_samples=30)
        # with torch.no_grad():
        #     log_probs *= log_probs
        # return -log_probs
        dist = self.forward(states)
        return dist.entropy()


    
    def to(self, device):
        """
        Moves the network to a device

        Parameters
        ----------
        device : torch.device
            The device to move the network to

        Returns
        -------
        nn.Module
            The current network, moved to a new device
        """
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        return super(Beta, self).to(device)    
