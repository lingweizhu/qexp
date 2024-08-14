from abc import ABC, abstractmethod
from jaxtyping import Float
import numpy as np
import torch
from .base_agent import BaseAgent
import torch.nn.functional as F

import sys
sys.path.append("../utils")
from utils.nn_utils import hard_update


class MPO(BaseAgent):
    """
    GreedyAC style implementation of MPO based on Martha's suggestion
    """
    def __init__(self,
                 discrete_action: bool,
                 action_dim: int,
                 state_dim: int,
                 gamma: float,
                 batch_size: float,
                 alpha: float,
                 device: torch.device,
                 actor: torch.nn.Module,
                 critic: torch.nn.Module,
                 replay_buffer: torch.nn.Module,
                 rho: float,
                 n_action_proposals: float,
                 ) -> None:
        super().__init__()
        self.discrete_action = discrete_action
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = device
        self.actor = actor
        self.critic = critic
        self.buffer = replay_buffer
        self.rho = rho
        self.n_action_proposals = n_action_proposals


    @torch.no_grad()
    def act(self, state: Float[np.ndarray, "state_dim"], greedy: bool=False) -> Float[np.ndarray, "action_dim"]:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, action_mean = self.actor.sample(state)
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
        next_state_action, _, _ = self.actor.policy.sample(next_state_batch)
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
        return q_loss.detach().item()

    def update_actor(self) -> None:
        """
        greedy-ac style best action sampling + KL trick update for MPO
        \min KL(\pi_{k+1} || \pi_{\theta}) = E_{\pi_{k+1}} [\ln \pi_{k+1} - \ln\pi_{\theta}]
        and \pi_{k+1} \propto \pi_{k}\exp((q-v)/lambda )

        therefore, the loss becomes: - E_{\pi_k} [\exp((q-v)/lambda)  \ln\pi_{\theta}] 
        we sample n_action_proposals best actions for both computing the expectation, q, and \ln\pi_{\theta}
        """    

        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.buffer.sample(batch_size=self.batch_size)
        if state_batch is None:
            # Too few samples in the buffer to sample
            return
        action_batch, _, _, = self.actor.policy.sample(state_batch, self.n_action_proposals)
        action_batch = action_batch.permute(1, 0, 2)
        action_batch = action_batch.reshape(self.batch_size * self.n_action_proposals, self.action_dim)
        stacked_s_batch = state_batch.repeat_interleave(self.n_action_proposals, dim=0)


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
        
        with torch.no_grad():
            best_q_values = self.critic.value_net(stacked_s_batch, best_actions)

        logprobs = self.actor.policy.log_prob(stacked_s_batch, best_actions)
        exp_scale = torch.exp((best_q_values - best_q_values.max(dim=1, keepdim=True)[0]) / self.alpha)            

        policy_loss = - torch.mean(exp_scale * logprobs)
        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()

    def reset(self) -> None:
        pass
