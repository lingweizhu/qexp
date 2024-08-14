from abc import ABC, abstractmethod
from jaxtyping import Float
import numpy as np
import torch
from .base_agent import BaseAgent
import torch.nn.functional as F
import gymnasium
import copy
import sys
sys.path.append("../utils")
from utils.nn_utils import hard_update


class TD3(BaseAgent):
    """
    referred to td3 cleanRL implementation: 
    https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py#L160
    """
    def __init__(self,
                 discrete_action: bool,
                 action_dim: int,
                 state_dim: int,
                 gamma: float,
                 batch_size: float,
                 device: torch.device,
                 actor: torch.nn.Module,
                 critic: torch.nn.Module,
                 replay_buffer: torch.nn.Module,
                 action_space: gymnasium.spaces,
                 exploration_noise: float,
                 policy_noise: float,
                 noise_clip: float,
                 ) -> None:
        super().__init__()
        self.discrete_action = discrete_action
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.actor = actor
        self.buffer = replay_buffer

        """critic has to be TD3Critic"""
        self.critic = critic
        self.actor_policy_target = copy.deepcopy(self.actor.policy)
        self.actor_policy_target.load_state_dict(self.actor.policy.state_dict())

        self.action_low = torch.FloatTensor(action_space.low)
        self.action_high = torch.FloatTensor(action_space.high)
        self.action_scale = torch.FloatTensor([(self.action_high - self.action_low) / 2.0])
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = 2
        self.policy_updates = 0


    @torch.no_grad()
    def act(self, state: Float[np.ndarray, "state_dim"]) -> Float[np.ndarray, "action_dim"]:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            _, _, action = self.actor.sample(state)
            action += torch.normal(0, self.action_scale * self.exploration_noise)
            act = action.detach().cpu().numpy()[0].clip(self.action_low.numpy(), self.action_high.numpy())

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

        with torch.no_grad():
            gauss_noise = torch.randn_like(action_batch, device=self.device) * self.policy_noise
            clipped_noise = gauss_noise.clamp(-self.noise_clip, self.noise_clip) * self.action_scale

            _, _, next_actions = self.actor_policy_target.sample(next_state_batch) 
            next_actions += clipped_noise
            next_state_actions = next_actions.clamp(self.action_low[0], self.action_high[0])
            
            qf1_next_target = self.critic.target_net_1(next_state_batch, next_state_actions)
            qf2_next_target = self.critic.target_net_2(next_state_batch, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        qf1_a_values = self.critic.value_net_1(state_batch, action_batch).view(-1)
        qf2_a_values = self.critic.value_net_2(state_batch, action_batch).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value.flatten())
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value.flatten())
        q_loss = qf1_loss + qf2_loss

        # next_state_action, _, _ = self.actor.policy.sample(next_state_batch)
        # with torch.no_grad():
        #     next_q = self.critic.target_net(next_state_batch, next_state_action)
        #     target_q_value = reward_batch + mask_batch * self.gamma * next_q
        # q_value = self.critic.value_net(state_batch, action_batch)
        # # Calculate the loss on the critic
        # # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # q_loss = F.mse_loss(target_q_value, q_value)
        # Update the critic
        self.critic.optimizer.zero_grad()
        q_loss.backward()
        self.critic.optimizer.step()
        return q_loss.detach().item()

    def update_actor(self) -> None:

        if self.policy_updates % self.policy_freq == 0:
            state_batch, _, _, _, _ = self.buffer.sample(batch_size=self.batch_size)
            if state_batch is None:
                return
            
            _, _, actions = self.actor.policy.sample(state_batch)
            actor_loss = -self.critic.value_net_1(state_batch, actions).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
        self.policy_updates += 1

    def reset(self) -> None:
        pass
