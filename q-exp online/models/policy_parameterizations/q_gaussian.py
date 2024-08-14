
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../utils")
from utils.nn_utils import weights_init_
from utils.qgaussian_utils import MultivariateBetaGaussianDiag, HeavyTailedBetaGaussian



class qMultivariateGaussian(nn.Module):
    # def __init__(self, entropic_index, num_actions, mean, shape, action_min, action_max):
    def __init__(self, num_inputs, num_actions, hidden_dim, activation, action_space, entropic_index, init=None):

        super(qMultivariateGaussian, self).__init__()

        self.entropic_index = entropic_index
        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_net = nn.Linear(hidden_dim, num_actions)
        self.log_shape_net = nn.Linear(hidden_dim, num_actions)

        self.action_max = torch.FloatTensor(action_space.high)
        self.action_min = torch.FloatTensor(action_space.low)

        self.apply(lambda module: weights_init_(module, init, activation))

        self.log_upper_bound = torch.log(self.action_max - 1e-6)
        self.log_lower_bound = torch.FloatTensor([-7.])

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")             
        # self.mvbg = MultivariateBetaGaussianDiag(self.mean, self.shape ** 2, alpha=self.entropic_index)

    def forward(self, state):
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))
        mean = torch.tanh(self.mean_net(x))
        mean = ((mean + 1) / 2) * (self.action_max - self.action_min) + self.action_min  # ∈ [action_min, action_max]
        shape = torch.exp(torch.clamp(self.log_shape_net(x), self.log_lower_bound, self.log_upper_bound))
        # shape = torch.exp(self.log_shape_net(x))

        return mean, shape
    
    def rsample(self, states, num_samples=1):
        mean, shape = self.forward(states)
        mvbg = MultivariateBetaGaussianDiag(mean, shape, alpha=self.entropic_index)
        actions = mvbg.rsample(sample_shape=(num_samples,))
        if num_samples == 1:
            actions = actions.squeeze(0)        
        actions = torch.clamp(actions, self.action_min, self.action_max)
        # qgauss_low, qgauss_high = mvbg.loc - mvbg._a, mvbg.loc + mvbg._a
        # actions = torch.clamp(actions, min=torch.max(self.action_min, qgauss_low), max=torch.min(qgauss_high, self.action_max))
        log_probs = mvbg.log_prob(actions).unsqueeze(-1)
        if self.num_actions == 1:
            log_probs.unsqueeze(-1)
        return actions, log_probs, mean
    
    def sample(self, states, num_samples=1, deterministic=False):
        mean, shape = self.forward(states)
        mvbg = MultivariateBetaGaussianDiag(mean, shape, alpha=self.entropic_index)
        actions = mvbg.sample((num_samples,))
        if num_samples == 1:
            actions = actions.squeeze(0)
        actions = torch.clamp(actions, self.action_min, self.action_max)            
        # qgauss_low, qgauss_high = mvbg.loc - mvbg._a, mvbg.loc + mvbg._a
        # actions = torch.clamp(actions, min=torch.max(self.action_min, qgauss_low), max=torch.min(qgauss_high, self.action_max))
        log_probs = mvbg.log_prob(actions).unsqueeze(-1)
        if self.num_actions == 1:
            log_probs.unsqueeze(-1)
        return actions, log_probs, mean
    
    def log_prob(self, states, actions):
        """
        q-Gaussian light-tailed can have zero probability somewhere
        and the distribution is changing.
        log-probing some actions from replay buffer can gives -inf
        replaces invalid entries with nearest samples in the current policy
        """
        mean, shape = self.forward(states)
        mvbg = MultivariateBetaGaussianDiag(mean, shape, alpha=self.entropic_index)
        actions = torch.clamp(actions, self.action_min, self.action_max)

        with torch.no_grad():
            given_log_probs = mvbg.log_prob(actions).unsqueeze(-1)

        supported = torch.where(~torch.isinf(given_log_probs))[0]
        support_actions = mvbg.sample(sample_shape=(30,))
        distance = torch.norm(support_actions - actions, dim=-1)
        min_distance = torch.min(distance, dim=0)[1]
        closest_actions = support_actions[min_distance, torch.arange(actions.size()[0]), :]
        closest_actions[supported] = actions[supported]
        log_probs = mvbg.log_prob(closest_actions).unsqueeze(-1)

        # if torch.any(torch.isnan(log_probs)) or torch.any(torch.isinf(log_probs)):
        #     log_probs[torch.isinf(log_probs)] = torch.nan
        #     log0_idx = np.where(torch.isnan(log_probs).sum(dim=-1).view(-1))[0]
        #     supported_actions = mvbg.rsample((30, )).detach()
        #     # print(f"sp action {supported_actions.shape}, action {actions.shape}, logprob {log_probs.shape}")
        #     supported_logprobs = mvbg.log_prob(supported_actions).unsqueeze(-1).reshape(-1, 1)
        #     # print(f"sp log prob {supported_logprobs.shape}")
        #     for idx in log0_idx:
        #         """
        #         closest in 2-norm sense
        #         """
        #         # print(f"selected actions {actions[idx, :]}")
        #         min_idx = torch.min(torch.squeeze(torch.nansum((actions[idx, :] - supported_actions.reshape(-1, 1))**2, dim=-1) ** 0.5), dim=0)[1]
        #         # print(f"min idx {min_idx}")
        #         log_probs[idx, :] = supported_logprobs[min_idx, :]

        if self.num_actions == 1:
            log_probs.unsqueeze(-1)        
        return log_probs
    
    def entropy(self, states):
        """GreedyAC style entropy computation"""
        _, log_probs, _ = self.rsample(states, num_samples=30)
        with torch.no_grad():
            log_probs *= log_probs
        return -log_probs
    

    def to(self, device):
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        return super(qMultivariateGaussian, self).to(device)
    










class qHeavyTailedGaussian(nn.Module):
    """
    this q-Gaussian is intended for heavy-tailed distribution
    implemented using Generalized Box-Muller Method
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, activation, action_space, entropic_index, init=None):

        super(qHeavyTailedGaussian, self).__init__()

        self.entropic_index = entropic_index
        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_net = nn.Linear(hidden_dim, num_actions)
        self.log_shape_net = nn.Linear(hidden_dim, num_actions)

        self.action_max = torch.FloatTensor(action_space.high)
        self.action_min = torch.FloatTensor(action_space.low)

        self.apply(lambda module: weights_init_(module, init, activation))

        # self.log_upper_bound = torch.log(self.action_max - 1e-6)
        self.log_lower_bound = torch.FloatTensor([-14.])

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")             

    def forward(self, state):
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))
        mean = torch.tanh(self.mean_net(x))
        mean = ((mean + 1) / 2) * (self.action_max - self.action_min) + self.action_min  # ∈ [action_min, action_max]
        shape = torch.exp(torch.clamp(self.log_shape_net(x), min=self.log_lower_bound))

        return mean, shape
    
    def rsample(self, states, num_samples=1):
        mean, shape = self.forward(states)
        hvbg = HeavyTailedBetaGaussian(mean, shape, entropic_index=self.entropic_index)
        actions = hvbg.rsample(sample_shape=(num_samples,))     
        if num_samples == 1:
            actions = actions.squeeze(0)
        actions = torch.clamp(actions, self.action_min, self.action_max)
        # qgauss_low, qgauss_high = mvbg.loc - mvbg._a, mvbg.loc + mvbg._a
        # actions = torch.clamp(actions, min=torch.max(self.action_min, qgauss_low), max=torch.min(qgauss_high, self.action_max))
        log_probs = hvbg.log_prob(actions)
        # print(f"action {actions.shape}, log prob {log_probs.shape}")
        return actions, log_probs, mean
    
    def sample(self, states, num_samples=1, deterministic=False):
        mean, shape = self.forward(states)
        hvbg = HeavyTailedBetaGaussian(mean, shape, entropic_index=self.entropic_index)
        actions = hvbg.sample((num_samples,))
        if num_samples == 1:
            actions = actions.squeeze(0)
        actions = torch.clamp(actions, self.action_min, self.action_max)            
        log_probs = hvbg.log_prob(actions)
        return actions, log_probs, mean
    
    def log_prob(self, states, actions):
        """actions can have"""
        mean, shape = self.forward(states)
        hvbg = HeavyTailedBetaGaussian(mean, shape, entropic_index=self.entropic_index)
        actions = torch.clamp(actions, self.action_min, self.action_max)
        log_probs = hvbg.log_prob(actions)
   
        return log_probs
    
    def entropy(self, states):
        """GreedyAC style entropy computation"""
        _, log_probs, _ = self.rsample(states, num_samples=30)
        with torch.no_grad():
            log_probs *= log_probs
        return -log_probs
    

    def to(self, device):
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        return super(qHeavyTailedGaussian, self).to(device)
