"""
pytorch implementation of q-Gaussians and reparametrized sampling
This code was adapted from the repo https://github.com/deep-spin/sparse_continuous_distributions/tree/main
MultivariateBetaGaussian: the authors in the paper above call the q-Gaussian as Beta-Gaussian
MultivariateBetaGaussianDiag is the policy class that forces the covariance matrix to be diagonal.
qMultivariateGaussian is the policy class you should call just like Gaussian/Beta/Softmax.

Note that unlike the Gaussian case, MultivariateBetaGaussianDiag does not mean that each dimension is independent of each other!!!
So to make the comparison to Gaussian diagonal covariance fair, 
I created another class UnivariateBetaGaussian and its policy class qUnivariateGaussian, that has each dimension independent.

sample usage:
from qGaussian import qMultivariateGaussian, qUnivariateGaussian

"""

import numpy as np

import numbers

import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.distributions import Beta
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property
from core.network.network_architectures import FCNetwork


LOG_2 = np.log(2)
LOG_PI = np.log(np.pi)


class _RealMatrix(constraints._Real):
    event_dim = 2


# reimplement scipy's cutoff for eigenvalues
def _eigvalsh_to_eps(spectra, cond=None, rcond=None):
    spectra = spectra.detach()
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = str(spectra.dtype)[-2:]
        factor = {'32': 1E3, '64': 1E6}
        cond = factor[t] * torch.finfo(spectra.dtype).eps

    return cond * torch.max(torch.abs(spectra), dim=-1).values


class FactorizedScale(object):
    def __init__(self, scales, cond=None, rcond=None, upper=True):
        """Factorized representation of a batch of scale PSD matrices."""

        self._zero = scales.new_zeros(size=(1,))

        # scales: [B x D x D]

        # s, u = torch.symeig(scales, eigenvectors=True, upper=True)
        s, u = torch.linalg.eigh(scales)

        eps = _eigvalsh_to_eps(s, cond, rcond)

        if torch.any(torch.min(s, dim=-1).values < -eps):
            raise ValueError('scale is not positive semidefinite')

        # probably could use searchsorted
        self.s_mask = s > eps.unsqueeze(dim=-1)

        self.u = u
        self.s = torch.where(self.s_mask, s, self._zero)
        self.s_inv = torch.where(self.s_mask, 1 / s, self._zero)

    @lazy_property
    def rank(self):
        return self.s_mask.sum(dim=-1)

    @lazy_property
    def trace(self):
        return self.s.sum(dim=-1)

    @lazy_property
    def trace_inv(self):
        return self.s_inv.sum(dim=-1)

    @lazy_property
    def log_det(self):
        log_s = torch.where(self.s_mask, torch.log(self.s), self._zero)
        return torch.sum(log_s, dim=-1)

    @lazy_property
    def log_det_inv(self):
        log_s_inv = torch.where(self.s_mask, torch.log(self.s_inv), self._zero)
        return torch.sum(log_s_inv, dim=-1)

    @lazy_property
    def L(self):
        return self.u @ torch.diag_embed(torch.sqrt(self.s))

    def L_mul_X(self, X):
        assert len(X.shape) > 1
        tmp = torch.sqrt(self.s).unsqueeze(-1) * X
        return self.u @ tmp

    @lazy_property
    def L_inv(self):
        return self.u @ torch.diag_embed(torch.sqrt(self.s_inv))

    def L_inv_t_mul_X(self, X):
        assert len(X.shape) > 1
        tmp = self.u.transpose(-2, -1) @ X
        return torch.sqrt(self.s_inv).unsqueeze(-1) * tmp


class DiagScale(FactorizedScale):
    def __init__(self, scales, cond=None, rcond=None, upper=True):
        """Compact representation of a batch of diagonal scale matrices."""

        self._zero = scales.new_zeros(size=(1,))

        eps = _eigvalsh_to_eps(scales, cond, rcond)

        if torch.any(torch.min(scales, dim=-1).values < -eps):
            raise ValueError('scale is not positive semidefinite')

        # probably could use searchsorted
        self.s_mask = scales > eps.unsqueeze(dim=-1)

        # self.u = torch.eye(scales.shape[-1])
        # print("d")
        # exit()
        self.s = torch.where(self.s_mask, scales, self._zero)
        self.s_inv = torch.where(self.s_mask, 1 / scales, self._zero)

    def L_mul_X(self, X):
        assert len(X.shape) > 1
        return torch.sqrt(self.s).unsqueeze(-1) * X

    def L_inv_t_mul_X(self, X):
        assert len(X.shape) > 1
        return torch.sqrt(self.s_inv).unsqueeze(-1) * X


class MultivariateBetaGaussian(Distribution):
    arg_constraints = {'loc': constraints.real_vector,
                       'scale': _RealMatrix(),
                       'alpha': constraints.greater_than(1)}
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc, scale=None, alpha=2, validate_args=None):
        """Batched multivariate beta-Gaussian random variable.

        The r.v. is parametrized in terms of a location (mean), scale
        (proportional to covariance) matrix, and scalar alpha>1.

        The pdf takes the form

        p(x) = [(alpha-1) * -.5 (x-u)' inv(Sigma) (x-u) - tau]_+ ** (alpha-1)

        where (u, Sigma) are the location and scale parameters.

        Parameters
        ----------
        loc: tensor, shape (broadcastable to) (*batch_dims, D)
            mean of the the distribution.

        scale: tensor, shape (broadcastable to) (*batch_dims, D, D)
            scale parameter Sigma of the distribution.

        alpha: scalar or tensor broadcastable to (*batch_dims)
            The exponent parameter of the distribution.
            For alpha -> 1, the distribution converges to a Gaussian.
            For alpha = 2, the distribution is a Truncated Paraboloid
                (n-d generalization of the Epanechnikov kernel.)
            For alpha -> infty, the distribution converges to a
            uniform on an ellipsoid.
        """

        if isinstance(alpha, numbers.Number):
            alpha = loc.new_tensor(alpha)

        # dimensions must be compatible to:
        # mean: [B x D]
        # scale: [B x D x D]
        # alpha: [B x 1]

        batch_shape = torch.broadcast_shapes(scale.shape[:-2],
                                             loc.shape[:-1],
                                             alpha.shape)

        event_shape = loc.shape[-1:]

        loc = loc.expand(batch_shape + (-1,))
        scale = scale.expand(batch_shape + (-1, -1))
        alpha = alpha.expand(batch_shape)

        self.loc = loc
        self.scale = scale
        self.alpha = alpha

        self._fact_scale = FactorizedScale(scale)

        super().__init__(batch_shape, event_shape, validate_args)

    @lazy_property
    def log_radius(self):
        """Logarithm of the max-radius R of the distribution."""

        alpha = self.alpha
        alpha_m1 = alpha - 1
        alpha_ratio = alpha / alpha_m1

        n = self._fact_scale.rank
        half_n = n / 2

        lg_n_a = torch.lgamma(half_n + alpha_ratio)
        lg_a = torch.lgamma(alpha_ratio)

        log_first = lg_n_a - lg_a - half_n * LOG_PI
        log_second = (LOG_2 - torch.log(alpha_m1)) / alpha_m1
        log_inner = log_first + log_second
        log_radius = (alpha_m1 / (2 + alpha_m1 * n)) * log_inner
        return log_radius

    @lazy_property
    def _tau(self):
        n = self._fact_scale.rank
        c = n + (2 / (self.alpha - 1))
        scaled_log_det = self._fact_scale.log_det / c
        return -torch.exp(2 * self.log_radius - LOG_2 - scaled_log_det)

    @lazy_property
    def _a(self):
        """Return the value a = |x-mu| where the density vanishes."""
        return torch.sqrt(-2 * self._tau.unsqueeze(-1) * self.scale) * 0.5

    @lazy_property
    def tsallis_entropy(self):
        """The Tsallis entropy -Omega_alpha of the distribution"""
        n = self._fact_scale.rank
        alpha_m1 = self.alpha - 1
        alpha_term = 1 / (self.alpha * alpha_m1)
        denom = 2 * self.alpha + n * alpha_m1
        tau_term = 2 * self._tau / denom
        return alpha_term + tau_term

    def _mahalanobis(self, x, broadcast_batch=False):

        # x: shape [B', D] -- possibly different B', right?
        # loc: shape [B, D].

        # 1. Mahalanobis term

        d = x.shape[-1]

        if broadcast_batch:
            # assume loc is [B, D] and manually insert ones
            # to make it [B, 1,...1, D]

            x_batch_shape = x.shape[:-1]
            x = x.reshape(x_batch_shape
                          + tuple(1 for _ in self.batch_shape)
                          + (d,))

        # these must be broadcastable to each other and end in d.

        # [B', B, D]
        diff = x - self.loc

        # right now with B=[], now, this yields [B', D]

        diff = diff.unsqueeze(dim=-1)
        # Li = self._fact_scale.L_inv
        # diff_scaled = (Li.transpose(-2, -1) @ diff).squeeze(dim=-1)
        diff_scaled = self._fact_scale.L_inv_t_mul_X(diff).squeeze(dim=-1)
        maha = diff_scaled.square().sum(dim=-1) / 2
        return maha

    def pdf(self, x, broadcast_batch=False):
        """Probability of an broadcastable observation x (could be zero)"""
        f = -self._tau - self._mahalanobis(x, broadcast_batch)
        return torch.clip((self.alpha-1) * f, min=0) ** (1 / (self.alpha-1))

    def log_prob(self, x, broadcast_batch=False):
        """Log-probability of an broadcastable observation x (could be -inf)"""
        return torch.log(self.pdf(x, broadcast_batch))

    def cross_fy(self, x, broadcast_batch=False):
        """The cross-Omega Fenchel-Young loss w.r.t. a Dirac observation x"""

        maha = self._mahalanobis(x, broadcast_batch)
        n = self._fact_scale.rank
        scaled_entropy = (1 + (n * (self.alpha - 1)) / 2) * self.tsallis_entropy
        return maha + scaled_entropy

    def _scale_when_sampling(self, LZ, sample_shape):
        # correction because Sigma is not Sigma_tilde
        n = self._fact_scale.rank
        c = torch.exp(-self._fact_scale.log_det / (2 * n + 4 / (self.alpha-1)))
        c = c.expand(sample_shape + c.shape).unsqueeze(-1)
        return c * LZ

    def rsample(self, sample_shape):
        """Draw samples from the distribution."""
        shape = self._extended_shape(sample_shape)
        # print(shape) if called with (5,) gives (5,2,3)

        radius = torch.exp(self.log_radius)
        radius = radius.expand(sample_shape + radius.shape)

        mask = self._fact_scale.s_mask.expand(shape)

        # project U onto the correct sphere)
        U = torch.randn(shape)
        U = torch.where(mask, U, U.new_zeros(1))
        norm = U.norm(dim=-1).unsqueeze(dim=-1)
        norm[norm==0] += 1e-6
        U /= norm

        n = self._fact_scale.rank
        half_n = n / 2
        alpha_m1 = self.alpha - 1
        alpha_ratio = self.alpha / alpha_m1

        ratio_dist = Beta(half_n, alpha_ratio).expand(shape[:-1])
        ratio = ratio_dist.rsample()
        r = radius * torch.sqrt(ratio)

        Z = r.unsqueeze(dim=-1) * U
        Z = Z.unsqueeze(dim=-1)

        # L = self._fact_scale.L
        # z @ Lt = (L @ Zt).t
        # LZ = (L @ Z).squeeze(dim=-1)
        LZ = self._fact_scale.L_mul_X(Z).squeeze(dim=-1)
        scaled_Z = self._scale_when_sampling(LZ, sample_shape)
        return self.loc + scaled_Z


class MultivariateBetaGaussianDiag(MultivariateBetaGaussian):
    arg_constraints = {'loc': constraints.real_vector,
                       'scale': constraints.greater_than_eq(0),
                       'alpha': constraints.greater_than(1)}

    def __init__(self, loc, scale=None, alpha=2, validate_args=None):
        """Batched multivariate beta-Gaussian random variable w/ diagonal scale.

        The r.v. is parametrized in terms of a location (mean), diagonal scale
        (proportional to covariance) matrix, and scalar alpha>1.

        The pdf takes the form

        p(x) = [(alpha-1) * -.5 (x-u)' inv(Sigma) (x-u) - tau]_+ ** (alpha-1)

        where (u, Sigma) are the location and scale parameters.

        Parameters
        ----------
        loc: tensor, shape (broadcastable to) (*batch_dims, D)
            mean of the the distribution.

        scale: tensor, shape (broadcastable to) (*batch_dims, D)
            diagonal of the scale parameter Sigma.

        alpha: scalar or tensor broadcastable to (*batch_dims)
            The exponent parameter of the distribution.
            For alpha -> 1, the distribution converges to a Gaussian.
            For alpha = 2, the distribution is a Truncated Paraboloid
                (n-d generalization of the Epanechnikov kernel.)
            For alpha -> infty, the distribution converges to a
            uniform on an ellipsoid.
        """

        if isinstance(alpha, numbers.Number):
            alpha = loc.new_tensor(alpha)

        batch_shape = torch.broadcast_shapes(scale.shape[:-1],
                                             loc.shape[:-1],
                                             alpha.shape)

        event_shape = loc.shape[-1:]

        loc = loc.expand(batch_shape + (-1,))
        scale = scale.expand(batch_shape + (-1,))
        alpha = alpha.expand(batch_shape)

        self.loc = loc
        self.alpha = alpha
        self.scale = scale

        self._fact_scale = DiagScale(scale)

        Distribution.__init__(self, batch_shape, event_shape, validate_args)


class qMultivariateGaussian(nn.Module):
    # def __init__(self, entropic_index, num_actions, mean, shape, action_min, action_max):
    def __init__(self, device, state_dim, num_actions, hidden_units, action_min, action_max, entropic_index):

        super(qMultivariateGaussian, self).__init__()

        self.entropic_index = entropic_index
        # self.entropic_index_net = FCNetwork(device, state_dim, hidden_units, 1,
        #                                 head_activation=lambda x: nn.functional.softplus(x)+torch.tensor(1.000001))

        self.num_actions = num_actions

        self.mean_net = FCNetwork(device, state_dim, hidden_units, num_actions) #nn.Parameter(torch.FloatTensor([mean]*num_actions), requires_grad=True)
        self.log_shape_net = FCNetwork(device, state_dim, hidden_units, num_actions) #nn.Parameter(torch.FloatTensor([shape]*num_actions), requires_grad=True)

        # self.mvbg = MultivariateBetaGaussianDiag(self.mean, self.shape ** 2, alpha=self.entropic_index)
        self.log_upper_bound = np.log(action_max - 1e-6)
        action_max = np.ones(num_actions) * (action_max - 1e-6)
        action_min = np.ones(num_actions) * (action_min + 1e-6)
        self.action_max = torch.FloatTensor(action_max)
        self.action_min = torch.FloatTensor(action_min)

    def forward(self, x):

        # mean = torch.clamp(self.mean, min=self.action_min, max=self.action_max)
        # shape = torch.clamp(self.shape, min=1e-6)
        mean = torch.tanh(self.mean_net(x))
        mean = torch.clamp(mean, -1. + 1e-6, 1. - 1e-6)
        mean = ((mean + 1) / 2) * (self.action_max - self.action_min) + self.action_min  # ∈ [action_min, action_max]
        shape = torch.exp(torch.clamp(self.log_shape_net(x), -14., self.log_upper_bound))
        return mean, shape
    
    def rsample(self, x, num_samples=1):
        mean, shape = self.forward(x)
        self.mvbg = MultivariateBetaGaussianDiag(mean, shape, alpha=self.entropic_index)
        # self.mvbg = MultivariateBetaGaussianDiag(mean, shape ** 2, alpha=self.entropic_index_net(x).squeeze(1))
        actions = self.mvbg.rsample(sample_shape=(num_samples,))
        if num_samples == 1:
            actions = actions.squeeze(0)        
        actions = torch.clamp(actions, self.action_min, self.action_max)
        log_probs = self.mvbg.log_prob(actions)
        # if len(torch.where(actions == 1)[0]) > 0 or len(torch.where(actions == -1)[0])>0:
        #     print("rsample: risky atanh", actions[torch.where(actions == 1)[0]])
        # print("rsam", mean.mean(dim=0), log_probs.mean())
        return actions, log_probs
    
    def sample(self, x, num_samples=1, deterministic=False):
        mean, shape = self.forward(x)
        self.mvbg = MultivariateBetaGaussianDiag(mean, shape, alpha=self.entropic_index)
        # with torch.no_grad():
        #     self.mvbg = MultivariateBetaGaussianDiag(mean, shape ** 2, alpha=self.entropic_index_net(x).squeeze(1))
        actions = self.mvbg.sample(sample_shape=(num_samples,)).detach()
        if num_samples == 1:
            actions = actions.squeeze(0)
        actions = torch.clamp(actions, self.action_min, self.action_max)
        with torch.no_grad():
            log_probs = self.mvbg.log_prob(actions)
        # print("samp", mean.mean(dim=0), log_probs.mean())
        return actions, log_probs
    
    def log_prob(self, observation, actions):
        mean, shape = self.forward(observation)
        self.mvbg = MultivariateBetaGaussianDiag(mean, shape, alpha=self.entropic_index)
        with torch.no_grad():
            given_action_logp = self.mvbg.log_prob(actions)
        supported = torch.where(~torch.isinf(given_action_logp))[0]
        support_actions = self.mvbg.sample(sample_shape=(100,)).detach()
        distance = torch.norm(support_actions - actions, dim=-1)
        min_distance = torch.min(distance, dim=0)[1]
        closest_actions = support_actions[min_distance, torch.arange(actions.size()[0]), :]
        closest_actions[supported] = actions[supported]
        logp = self.mvbg.log_prob(closest_actions)
        return logp
    

    def to(self, device):
        self.action_max = self.action_max.to(device)
        self.action_min = self.action_min.to(device)
        return super(qMultivariateGaussian, self).to(device)


"""
I implemented HeavyTailed qGaussian using the generalized Box-Muller method
the naming BetaGaussian is just to follow the convention. 
However, this beta is different from the one above, 
 - here beta = q, use 1 < entropic_index < 3, heavy tailed
 - above beta = 2-q, alpha > 0   ->  q < 1

If you are confused, use q=2 for this heavy-tailed class, it always work
"""


class HeavyTailedBetaGaussian(Distribution):
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "entropic_index": constraints.interval(1, 3)
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, entropic_index=2, validate_args=None):

        if isinstance(entropic_index, numbers.Number):
            entropic_index = loc.new_tensor(entropic_index)

        batch_shape = torch.broadcast_shapes(scale.shape[:-1], loc.shape[:-1], entropic_index.shape)
        event_shape = loc.shape[-1:]

        loc = loc.expand(batch_shape + (-1,))
        scale = scale.expand(batch_shape + (-1,))
        # entropic_index = entropic_index.expand(batch_shape)

        self.loc = loc
        self.scale = scale
        self.entropic_index = torch.FloatTensor(entropic_index)

        self._fact_scale = DiagScale(scale)

        super().__init__(batch_shape, event_shape, validate_args)

    # @lazy_property
    # def _normalization(self):
    #     q = self.entropic_index
    #     ratio = (3 - q) / (2 * (q - 1))
    #     Z = torch.exp(torch.lgamma(ratio)) * (torch.pi * (3 - q)) ** 0.5 / (
    #                 torch.exp(torch.lgamma(ratio + 0.5)) * (q - 1) ** 0.5) * (
    #                     0.5 * self._fact_scale.log_det).exp()
    #     return Z
    @lazy_property
    def _normalization(self):
        n = self._fact_scale.rank
        q = self.entropic_index
        ratio = (3-q) / (2*(q-1))
        Z =  torch.exp(torch.lgamma(ratio)) * ((torch.pi * (3-q)/(q-1)) ** (n/2)) / (torch.exp(torch.lgamma(ratio + 0.5*n))) * (0.5*self._fact_scale.log_det).exp()
        return Z

    def _mahalanobis(self, x, broadcast_batch=False):

        d = x.shape[-1]

        if broadcast_batch:
            # assume loc is [B, D] and manually insert ones
            # to make it [B, 1,...1, D]

            x_batch_shape = x.shape[:-1]
            x = x.reshape(x_batch_shape
                          + tuple(1 for _ in self.batch_shape)
                          + (d,))

        # [B', B, D]
        diff = x - self.loc

        diff = diff.unsqueeze(dim=-1)
        diff_scaled = self._fact_scale.L_inv_t_mul_X(diff).squeeze(dim=-1)
        maha = diff_scaled.square().sum(dim=-1) / 2
        return maha

    def _exp_q(self, inputs):
        return torch.maximum(torch.FloatTensor([0.]), 1 + (1 - self.entropic_index) * inputs) ** (
                    1 / (1 - self.entropic_index))

    def _log_q(self, inputs, q):
        if q == 1:
            return torch.log(inputs)
        else:
            return (inputs ** (1 - q) - 1) / (1 - q)

    def pdf(self, x, broadcast_batch=False):
        f = - self._mahalanobis(x, broadcast_batch)
        return self._exp_q(f / (3 - self.entropic_index)) / self._normalization

    def log_prob(self, x, broadcast_batch=False):
        return torch.log(self.pdf(x, broadcast_batch)).unsqueeze(dim=-1)

    def rsample(self, sample_shape):
        """
        Generalized Box Muller Method:
        https://ieeexplore.ieee.org/document/4385787
        """
        batch_size = self.loc.shape[0]
        event_shape = self.loc.shape[1]
        batch_shape = sample_shape[0]
        shape = (batch_shape, batch_size, event_shape, 2)

        u = torch.rand(shape)

        """known q', want to produce q'-Gaussian"""
        q = (self.entropic_index - 1) / (3 - self.entropic_index)
        z1 = torch.sqrt(-2 * self._log_q(u[:, :, :, 0], q=q)) * torch.cos(2 * torch.pi * u[:, :, :, 1])

        sample = self.loc + torch.sqrt(self.scale) * z1
        return sample


class qHeavyTailedGaussian(nn.Module):
    """
    this q-Gaussian is intended for heavy-tailed distribution
    implemented using Generalized Box-Muller Method
    """

    # def __init__(self, device, num_inputs, num_actions, hidden_dim, activation, action_min, action_max,
    #              entropic_index, init=None):
    def __init__(self, device, state_dim, num_actions, hidden_units, action_min, action_max, entropic_index):

        super(qHeavyTailedGaussian, self).__init__()

        self.entropic_index = entropic_index
        self.num_actions = num_actions

        # self.linear1 = nn.Linear(num_inputs, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        #
        # self.mean_net = nn.Linear(hidden_dim, num_actions)
        # self.log_shape_net = nn.Linear(hidden_dim, num_actions)

        self.mean_net = FCNetwork(device, state_dim, hidden_units, num_actions) #nn.Parameter(torch.FloatTensor([mean]*num_actions), requires_grad=True)
        self.log_shape_net = FCNetwork(device, state_dim, hidden_units, num_actions) #nn.Parameter(torch.FloatTensor([shape]*num_actions), requires_grad=True)

        action_max = np.ones(num_actions) * (action_max - 1e-6)
        action_min = np.ones(num_actions) * (action_min + 1e-6)
        self.action_max = torch.FloatTensor(action_max)
        self.action_min = torch.FloatTensor(action_min)

        # self.apply(lambda module: nn.init.xavier_uniform_(module, init, activation))

        self.log_upper_bound = np.log(1000)
        self.log_lower_bound = -self.log_upper_bound

        # if activation == "relu":
        #     self.act = torch.nn.functional.relu
        # elif activation == "tanh":
        #     self.act = torch.tanh
        # else:
        #     raise ValueError(f"unknown activation {activation}")

        self.to(device)

    def forward(self, state):
        # x = self.act(self.linear1(state))
        # x = self.act(self.linear2(x))
        # mean = torch.tanh(self.mean_net(x))
        mean = torch.tanh(self.mean_net(state))
        mean = ((mean + 1) / 2) * (
                    self.action_max - self.action_min) + self.action_min  # ∈ [action_min, action_max]
        # shape = torch.exp(
        #     torch.clamp(self.log_shape_net(x), self.log_lower_bound, self.log_upper_bound))
        shape = torch.exp(
            torch.clamp(self.log_shape_net(state), self.log_lower_bound, self.log_upper_bound))

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
        # print(f"outer rsample, action {actions.shape}, log prob {log_probs.shape}")
        return actions, log_probs

    def sample(self, states, num_samples=1, deterministic=False):
        mean, shape = self.forward(states)
        hvbg = HeavyTailedBetaGaussian(mean, shape, entropic_index=self.entropic_index)
        actions = hvbg.sample((num_samples,))
        if num_samples == 1:
            actions = actions.squeeze(0)
        actions = torch.clamp(actions, self.action_min, self.action_max)
        log_probs = hvbg.log_prob(actions)
        # print(f"outer sample, action {actions.shape}, log prob {log_probs.shape}")
        return actions, log_probs

    def log_prob(self, states, actions):
        """actions can have"""
        mean, shape = self.forward(states)
        hvbg = HeavyTailedBetaGaussian(mean, shape, entropic_index=self.entropic_index)
        actions = torch.clamp(actions, self.action_min, self.action_max)
        log_probs = hvbg.log_prob(actions)
        # if torch.any(torch.isnan(log_probs)) or torch.any(torch.isinf(log_probs)):
        #     from IPython import embed; embed()
        #     raise RuntimeError("NaN detected in log probs!")
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

    def distribution(self, x, dim=-1):
        if dim == -1:
            with torch.no_grad():
                mean, shape = self.forward(x)
                dist = HeavyTailedBetaGaussian(mean, shape, entropic_index=self.entropic_index)
            return dist, mean, shape, None
        else:
            with torch.no_grad():
                mean, shape = self.forward(x)
                mean = mean[:, dim: dim+1]
                shape = shape[:, dim: dim+1]
                dist = HeavyTailedBetaGaussian(mean, shape, entropic_index=self.entropic_index)
            return dist, mean, shape, None

