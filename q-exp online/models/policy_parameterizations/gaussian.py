import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import math

import sys
sys.path.append("../../utils")
from utils.nn_utils import weights_init_


class Gaussian(nn.Module):
    """
    Class Gaussian implements a policy following Gaussian distribution
    in each state, parameterized as an MLP. The predicted mean is scaled to be
    within `(action_min, action_max)` using a `tanh` activation.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, activation,
                 action_space, clip_stddev=1000, init=None):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            The number of elements in the state feature vector
        num_actions : int
            The dimensionality of the action vector
        hidden_dim : int
            The number of units in each hidden layer of the network
        action_space : gym.spaces.Space
            The action space of the environment
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        """
        super(Gaussian, self).__init__()

        self.num_actions = num_actions

        # Determine standard deviation clipping
        self.clip_stddev = clip_stddev > 0
        self.clip_std_threshold = np.log(clip_stddev)

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        # Initialize weights
        self.apply(lambda module: weights_init_(module, init, activation))

        # Action rescaling
        self.action_max = torch.FloatTensor(action_space.high)
        self.action_min = torch.FloatTensor(action_space.low)

        self.last_mean = 0
        self.last_std = 1

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state):
        """
        Performs the forward pass through the network, predicting the mean
        and the log standard deviation.

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        2-tuple of torch.Tensor of float
            The mean and log standard deviation of the Gaussian policy in the
            argument state
        """
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))

        mean = torch.tanh(self.mean_linear(x))
        mean = ((mean + 1) / 2) * (self.action_max - self.action_min) + \
            self.action_min  # ∈ [action_min, action_max]
        log_std = self.log_std_linear(x)

        # Works better with std dev clipping to ±1000
        if self.clip_stddev:
            log_std = torch.clamp(log_std, min=-self.clip_std_threshold,
                                  max=self.clip_std_threshold)
        return mean, log_std

    def rsample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        # For re-parameterization trick (mean + std * N(0,1))
        # rsample() implements the re-parameterization trick
        action = normal.rsample((num_samples,))
        action = torch.clamp(action, self.action_min, self.action_max)
        if num_samples == 1:
            action = action.squeeze(0)

        log_prob = normal.log_prob(action)
        if self.num_actions == 1:
            log_prob.unsqueeze(-1)

        self.last_mean = mean
        self.last_std = std

        return action, log_prob, mean

    def sample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state

        Parameters
        ----------
        state : torch.Tensor of float
             The input state to predict the policy in
        num_samples : int
            The number of actions to sample

        Returns
        -------
        torch.Tensor of float
            A sampled action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        # Non-differentiable
        action = normal.sample((num_samples,))
        action = torch.clamp(action, self.action_min, self.action_max)

        if num_samples == 1:
            action = action.squeeze(0)

        log_prob = normal.log_prob(action)
        if self.num_actions == 1:
            log_prob.unsqueeze(-1)

        self.last_mean = mean
        self.last_std = std

        return action, log_prob, mean

    def log_prob(self, states, actions, show=False):
        """
        Returns the log probability of taking actions in states. The
        log probability is returned for each action dimension
        separately, and should be added together to get the final
        log probability
        """
        mean, log_std = self.forward(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        if self.num_actions > 1:
            normal = Independent(normal, 1)

        log_prob = normal.log_prob(actions)
        if self.num_actions == 1:
            log_prob.unsqueeze(-1)

        if show:
            print(torch.cat([mean, std], axis=1)[0])

        return log_prob

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
        return super(Gaussian, self).to(device)



    def entropy(self, states):
        # mean, log_std = [param.detach() for param in self.forward()]
        _, log_std = self.forward(states)
        return 0.5 + 0.5 * math.log(2 * math.pi) + log_std

    def get_stats(self) -> list:
        if isinstance(self.last_mean, torch.Tensor):
            return [self.last_mean.squeeze().tolist(), self.last_std.squeeze().tolist()]
        else:
            return [[self.last_mean], [self.last_std]]
