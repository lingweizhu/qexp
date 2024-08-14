import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import sys
sys.path.append("../../utils")
from utils.nn_utils import weights_init_


# Global variables
EPSILON = 1e-6


class SquashedGaussian(nn.Module):
    """
    Class SquashedGaussian implements a policy following a squashed
    Gaussian distribution in each state, parameterized by an MLP.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, activation,
                 action_space=None, clip_stddev=1000, init=None):
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
        activation : str
            The activation function to use, one of 'relu', 'tanh'
        action_space : gym.spaces.Space, optional
            The action space of the environment, by default None. This argument
            is used to ensure that the actions are within the correct scale.
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
        super(SquashedGaussian, self).__init__()

        self.num_actions = num_actions

        # Determine standard deviation clipping
        self.clip_stddev = clip_stddev > 0
        self.clip_std_threshold = np.log(clip_stddev)

        # Set up the layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        # Initialize weights
        self.apply(lambda module: weights_init_(module, init, activation))

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation function {activation}")

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

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        if self.clip_stddev:
            log_std = torch.clamp(log_std, min=-self.clip_std_threshold,
                                  max=self.clip_std_threshold)
        return mean, log_std

    def sample(self, state, num_samples=1):
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

        x_t = normal.sample((num_samples,))
        if num_samples == 1:
            x_t = x_t.squeeze(0)
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +
                              EPSILON).sum(axis=-1).reshape(log_prob.shape)
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

    def rsample(self, state, num_samples=1):
        """
        Samples the policy for an action in the argument state using
        the reparameterization trick

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
        x_t = normal.rsample((num_samples,))
        if num_samples == 1:
            x_t = x_t.squeeze(0)
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +
                              EPSILON).sum(axis=-1).reshape(log_prob.shape)
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

    def log_prob(self, state_batch, x_t_batch):
        """
        Calculates the log probability of taking the action generated
        from x_t, where x_t is returned from sample or rsample. The
        log probability is returned for each action dimension separately.
        """
        mean, log_std = self.forward(state_batch)
        std = log_std.exp()
        normal = Normal(mean, std)

        if self.num_actions > 1:
            normal = Independent(normal, 1)

        y_t = torch.tanh(x_t_batch)
        log_prob = normal.log_prob(x_t_batch)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +
                              EPSILON).sum(axis=-1).reshape(log_prob.shape)
        if self.num_actions > 1:
            log_prob = log_prob.unsqueeze(-1)

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
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(SquashedGaussian, self).to(device)
