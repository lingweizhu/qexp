from abc import ABC, abstractmethod
from jaxtyping import Float
import numpy as np
import torch
from .base_agent import BaseAgent
import torch.nn.functional as F

import sys
sys.path.append("../utils")
from utils.nn_utils import hard_update


class GreedyAC(BaseAgent):
    def __init__(self,
                 discrete_action: bool,
                 action_dim: int,
                 state_dim: int,
                 gamma: float,
                 batch_size: float,
                 alpha: float,
                 device: torch.device,
                 behavior_policy: torch.nn.Module,
                 proposal_policy: torch.nn.Module,
                 critic: torch.nn.Module,
                 replay_buffer: torch.nn.Module,
                 rho: float,
                 n_action_proposals: float,
                 entropy_from_single_sample: bool) -> None:
        super().__init__()
        self.discrete_action = discrete_action
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = device
        self.bp = behavior_policy
        self.pp = proposal_policy
        self.critic = critic
        self.buffer = replay_buffer
        self.rho = rho
        self.n_action_proposals = n_action_proposals
        self.entropy_from_single_sample = entropy_from_single_sample
        # match the initialization of both policies
        hard_update(self.bp.policy, self.pp.policy)

    @torch.no_grad()
    def act(self, state: Float[np.ndarray, "state_dim"], greedy: bool=False) -> Float[np.ndarray, "action_dim"]:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, action_mean = self.bp.sample(state)
        act = action.detach().cpu().numpy()[0]
        if greedy:
            act = action_mean.detach().cpu().numpy()[0]
        if not self.discrete_action:
            return act
        else:
            return int(act[0])

    def update_critic(self) -> float:
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.buffer.sample(batch_size=self.batch_size)
        if state_batch is None:
            # Too few samples in the buffer to sample
            return
        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        next_state_action, _, _ = self.bp.policy.sample(next_state_batch)
        with torch.no_grad():
            next_q = self.critic.target_net(next_state_batch, next_state_action)
            target_q_value = reward_batch + mask_batch * self.gamma * next_q
        q_value = self.critic.value_net(state_batch, action_batch)
        # Calculate the loss on the critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q_loss = F.mse_loss(target_q_value, q_value)
        # Update the critic
        self.critic.optimizer.zero_grad()
        q_loss.backward()
        self.critic.optimizer.step()
        #from IPython import embed; embed(); exit()
        return q_loss.detach().item()

    def update_actor(self) -> None:
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.buffer.sample(batch_size=self.batch_size)
        if state_batch is None:
            # Too few samples in the buffer to sample
            return
        # sample action proposals
        action_batch, _, _, = self.pp.policy.sample(state_batch, self.n_action_proposals)
        action_batch = action_batch.permute(1, 0, 2)
        action_batch = action_batch.reshape(self.batch_size * self.n_action_proposals, self.action_dim)
        stacked_s_batch = state_batch.repeat_interleave(self.n_action_proposals, dim=0)
        # -------- behavior policy ------- #
        # Get the values of the sampled actions and find the best rho * n_action_proposals actions
        with torch.no_grad():
            q_values = self.critic.value_net(stacked_s_batch, action_batch)
        q_values = q_values.reshape(self.batch_size, self.n_action_proposals, 1)
        sorted_q = torch.argsort(q_values, dim=1, descending=True)
        best_ind = sorted_q[:, :int(self.rho * self.n_action_proposals)]
        best_ind = best_ind.repeat_interleave(self.action_dim, -1)
        action_batch = action_batch.reshape(self.batch_size, self.n_action_proposals, self.action_dim)
        best_actions = torch.gather(action_batch, 1, best_ind)
        # Reshape samples for calculating the loss
        samples = int(self.rho * self.n_action_proposals)
        stacked_s_batch = state_batch.repeat_interleave(samples, dim=0)
        best_actions = torch.reshape(best_actions, (-1, self.action_dim))
        # compute behavior policy loss
        behavior_loss = self.bp.policy.log_prob(stacked_s_batch, best_actions)
        behavior_loss = -behavior_loss.mean()
        # update behavior policy
        self.bp.optimizer.zero_grad()
        behavior_loss.backward()
        self.bp.optimizer.step()
        # -------- proposal policy ------- #
        # Calculate entropy for proposal policy
        stacked_s_batch = state_batch.repeat_interleave(self.n_action_proposals, dim=0)
        stacked_s_batch = stacked_s_batch.reshape(-1, self.state_dim)
        action_batch = action_batch.reshape(-1, self.action_dim)
        proposal_entropy = self.pp.policy.log_prob(stacked_s_batch, action_batch)
        with torch.no_grad():
            proposal_entropy *= proposal_entropy
        proposal_entropy = proposal_entropy.reshape(self.batch_size, self.n_action_proposals, 1)
        if self.entropy_from_single_sample:
            proposal_entropy = -proposal_entropy[:, 0, :]
        else:
            proposal_entropy = -proposal_entropy.mean(axis=1)
        # Calculate proposal loss
        stacked_s_batch = state_batch.repeat_interleave(samples, dim=0)
        proposal_loss = self.pp.policy.log_prob(stacked_s_batch, best_actions)
        proposal_loss = proposal_loss.reshape(self.batch_size, samples, 1)
        proposal_loss = proposal_loss.mean(axis=1)
        proposal_loss = proposal_loss + (proposal_entropy * self.alpha)
        proposal_loss = -proposal_loss.mean()
        # Update the proposal policy
        self.pp.optimizer.zero_grad()
        proposal_loss.backward()
        self.pp.optimizer.step()


    def reset(self) -> None:
        pass
