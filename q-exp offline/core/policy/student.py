import math
import numpy as np
import torch
import torch.nn as nn
from torch import inf, nan
from torch.distributions import Chi2, constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, broadcast_all

from core.network.network_architectures import FCNetwork

__all__ = ["StudentT"]


class StudentT(Distribution):
    r"""
    Creates a Student's t-distribution parameterized by degree of
    freedom :attr:`df`, mean :attr:`loc` and scale :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = StudentT(torch.tensor([2.0]))
        >>> m.sample()  # Student's t-distributed with degrees of freedom=2
        tensor([ 0.1046])

    Args:
        df (float or Tensor): degrees of freedom
        loc (float or Tensor): mean of the distribution
        scale (float or Tensor): scale of the distribution
    """
    arg_constraints = {
        "df": constraints.positive,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        m = self.loc.clone(memory_format=torch.contiguous_format)
        m[self.df <= 1] = nan
        return m

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        m = self.df.clone(memory_format=torch.contiguous_format)
        m[self.df > 2] = (
            self.scale[self.df > 2].pow(2)
            * self.df[self.df > 2]
            / (self.df[self.df > 2] - 2)
        )
        m[(self.df <= 2) & (self.df > 1)] = inf
        m[self.df <= 1] = nan
        return m

    def __init__(self, df, loc=0.0, scale=1.0, validate_args=None):
        self.df, self.loc, self.scale = broadcast_all(df, loc, scale)
        self._chi2 = Chi2(self.df)
        batch_shape = self.df.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(StudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        new.df = self.df.expand(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new._chi2 = self._chi2.expand(batch_shape)
        super(StudentT, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        # NOTE: This does not agree with scipy implementation as much as other distributions.
        # (see https://github.com/fritzo/notebooks/blob/master/debug-student-t.ipynb). Using DoubleTensor
        # parameters seems to help.

        #   X ~ Normal(0, 1)
        #   Z ~ Chi2(df)
        #   Y = X / sqrt(Z / df) ~ StudentT(self.df)
        shape = self._extended_shape(sample_shape)
        X = _standard_normal(shape, dtype=self.df.dtype, device=self.df.device)
        Z = self._chi2.rsample(sample_shape)
        Y = X * torch.rsqrt(Z / self.df)
        return self.loc + self.scale * Y

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y = (value - self.loc) / self.scale
        Z = (
            self.scale.log()
            + 0.5 * self.df.log()
            + 0.5 * math.log(math.pi)
            + torch.lgamma(0.5 * self.df)
            - torch.lgamma(0.5 * (self.df + 1.0))
        )
        return -0.5 * (self.df + 1.0) * torch.log1p(y**2.0 / self.df) - Z

    def entropy(self):
        lbeta = (
            torch.lgamma(0.5 * self.df)
            + math.lgamma(0.5)
            - torch.lgamma(0.5 * (self.df + 1))
        )
        return (
            self.scale.log()
            + 0.5
            * (self.df + 1)
            * (torch.digamma(0.5 * (self.df + 1)) - torch.digamma(0.5 * self.df))
            + 0.5 * self.df.log()
            + lbeta
        )
    
class Student(nn.Module):
    # def __init__(self, num_actions, df, action_min, action_max, mean=0.0, shape=1.0):
    def __init__(self, device, state_dim, num_actions, hidden_units, action_min, action_max, df):

        super(Student, self).__init__()

        self.num_actions = num_actions

        """consider making degree of freedom: df learnable as well!"""
        # self.df = nn.Parameter(torch.FloatTensor([df]*num_actions), requires_grad=True)
        # self.df = torch.FloatTensor([df]*num_actions) #df
        self.df_net = FCNetwork(device, state_dim, hidden_units, num_actions,
                            head_activation=lambda x: nn.functional.softplus(x)+torch.tensor(1.000001))

        self.mean_net = FCNetwork(device, state_dim, hidden_units, num_actions) #nn.Parameter(torch.FloatTensor([mean]*num_actions), requires_grad=True)
        self.log_shape_net = FCNetwork(device, state_dim, hidden_units, num_actions) #nn.Parameter(torch.FloatTensor([shape]*num_actions), requires_grad=True)

        # self.student = StudentT(self.df, self.mean, self.shape ** 2)

        self.log_upper_bound = np.log(action_max - 1e-6)
        action_max = np.ones(num_actions) * (action_max - 1e-6)
        action_min = np.ones(num_actions) * (action_min + 1e-6)
        self.action_max = torch.FloatTensor(action_max)
        self.action_min = torch.FloatTensor(action_min)


    def forward(self, x):
        mean = torch.tanh(self.mean_net(x))
        mean = ((mean + 1) / 2) * (self.action_max - self.action_min) + self.action_min  # ∈ [action_min, action_max]
        # shape = self.shape_net(x)
        shape = torch.exp(torch.clamp(self.log_shape_net(x), -14., self.log_upper_bound))
        dfx = self.df_net(x)
        # dfx = torch.clamp(dfx, 0, 20)
        return mean, shape, dfx
    
    def rsample(self, x, num_samples=1):
        mean, shape, dfx = self.forward(x)
        self.student = StudentT(dfx, mean, shape ** 2)
        actions = self.student.rsample(sample_shape=(num_samples,))
        if num_samples == 1:
            actions = actions.squeeze(0)
        actions = torch.clamp(actions, self.action_min, self.action_max)
        log_probs = self.student.log_prob(actions)
        return actions, log_probs.mean(dim=-1, keepdim=True) #, (mean, shape), None
    
    def sample(self, x, num_samples=1, deterministic=False):
        with torch.no_grad():
            mean, shape, dfx = self.forward(x)
            self.student = StudentT(dfx, mean, shape ** 2)
        actions = self.student.sample(sample_shape=(num_samples,)).detach()
        if num_samples == 1:
            actions = actions.squeeze(0)
        actions = torch.clamp(actions, self.action_min, self.action_max)
        with torch.no_grad():
            log_probs = self.student.log_prob(actions)
        return actions, log_probs.mean(dim=-1, keepdim=True) #, (mean, shape), None
    
    def log_prob(self, state, actions):
        actions = torch.clamp(actions, self.action_min, self.action_max)
        mean, shape, dfx = self.forward(state)
        self.student = StudentT(dfx, mean, shape ** 2)
        return self.student.log_prob(actions).mean(dim=-1, keepdim=True)
    

    def to(self, device):
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        return super(Student, self).to(device)

    def distribution(self, x, dim=-1):
        if dim == -1:
            with torch.no_grad():
                mean, shape, dfx = self.forward(x)
                dist = StudentT(dfx, mean, shape ** 2)
            return dist, mean, shape, dfx
        else:
            with torch.no_grad():
                mean, shape, dfx = self.forward(x)
                mean = mean[:, dim: dim+1]
                shape = shape[:, dim: dim+1]
                dfx = dfx[:, dim: dim+1]
                dist = StudentT(dfx, mean, shape ** 2)
            return dist, mean, shape, dfx

class StudentFixDF(Student):
    def __init__(self, device, state_dim, num_actions, hidden_units, action_min, action_max, df):
        super(StudentFixDF, self).__init__(device, state_dim, num_actions, hidden_units, action_min, action_max, df)
        self.df = torch.FloatTensor([df]*num_actions)

    def forward(self, x):
        mean = torch.tanh(self.mean_net(x))
        mean = ((mean + 1) / 2) * (
                    self.action_max - self.action_min) + self.action_min  # ∈ [action_min, action_max]
        # shape = self.shape_net(x)
        shape = torch.exp(torch.clamp(self.log_shape_net(x), -14., self.log_upper_bound))
        return mean, shape, self.df
