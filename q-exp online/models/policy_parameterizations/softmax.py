import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import sys
sys.path.append("../../utils")
from utils.nn_utils import weights_init_


class Softmax(nn.Module):
    """
    Softmax implements a softmax policy in each state, parameterized
    using an MLP to predict logits.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, activation,
                 init=None):
        super(Softmax, self).__init__()

        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        # self.apply(weights_init_)
        self.apply(lambda module: weights_init_(module, init, activation))

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state):
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))
        return self.linear3(x)

    def sample(self, state, num_samples=1):
        logits = self.forward(state)

        if len(logits.shape) != 1 and (len(logits.shape) != 2 and 1 not in
           logits.shape):
            shape = logits.shape
            raise ValueError(f"expected a vector of logits, got shape {shape}")

        probs = F.softmax(logits, dim=1)

        policy = torch.distributions.Categorical(probs)
        actions = policy.sample((num_samples,))

        log_prob = F.log_softmax(logits, dim=1)

        log_prob = torch.gather(log_prob, dim=1, index=actions)
        if num_samples == 1:
            actions = actions.squeeze(0)
            log_prob = log_prob.squeeze(0)

        actions = actions.unsqueeze(-1)
        log_prob = log_prob.unsqueeze(-1)

        # return actions.float(), log_prob, None
        return actions.int(), log_prob, logits.argmax(dim=-1)

    def all_log_prob(self, states):
        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=1)

        return log_probs

    def log_prob(self, states, actions):
        """
        Returns the log probability of taking actions in states.
        """
        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=1)
        log_probs = torch.gather(log_probs, dim=1, index=actions.long())

        return log_probs

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
        return super(Softmax, self).to(device)
