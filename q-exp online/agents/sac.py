from abc import ABC, abstractmethod
from jaxtyping import Float
import numpy as np
import torch
from .base_agent import BaseAgent
import torch.nn.functional as F
from torch.optim import Adam

import sys
sys.path.append("../utils")
from utils.nn_utils import hard_update


class SAC(BaseAgent):
    def __init__(self,
                 action_space: object,
                 gamma: float,
                 batch_size: float,
                 alpha: float,
                 alpha_lr: float,
                 device: torch.device,
                 actor: torch.nn.Module,
                 critic: torch.nn.Module,
                 replay_buffer: torch.nn.Module,
                 baseline_actions: int = -1,
                 reparameterized: bool = True,
                 soft_q: bool = True,
                 double_q: bool = True,
                 n_samples_for_entropy: int = 1,
                 automatic_entropy_tuning: bool = False) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.buffer = replay_buffer

        if reparameterized and n_samples_for_entropy != 1:
            raise ValueError

        self._action_space = action_space
        self._baseline_actions = baseline_actions

        # Random hypers and fields
        self._gamma = gamma
        self._reparameterized = reparameterized
        self._soft_q = soft_q
        self._double_q = double_q
        if n_samples_for_entropy < 1:
            raise ValueError("cannot have n_samples_for_entropy < 1")
        self._num_samples = n_samples_for_entropy # Sample for likelihood-based gradient

        self._device = device

        # Experience replay buffer
        self._batch_size = batch_size

        # Automatic entropy tuning
        self._automatic_entropy_tuning = automatic_entropy_tuning
        self._alpha_lr = alpha_lr

        if self._automatic_entropy_tuning and self._alpha_lr <= 0:
            raise ValueError("should not use entropy lr <= 0")

        # Set up auto entropy tuning
        if self._automatic_entropy_tuning:
            self._target_entropy = -torch.prod(
                torch.Tensor(action_space.shape).to(self._device)
            ).item()
            self._log_alpha = torch.zeros(
                1,
                requires_grad=True,
                device=self._device,
            )
            self._alpha = self._log_alpha.exp().detach()
            self._alpha_optim = Adam([self._log_alpha], lr=self._alpha_lr)
        else:
            self._alpha = alpha  # Entropy scale

    @torch.no_grad()
    def act(
        self,
        state: Float[np.ndarray, "state_dim"],
        greedy: bool=False
    ) -> Float[np.ndarray, "action_dim"]:
        """
        Take an action given the state
        greedy: take a greedy action if True
        """
        state = torch.FloatTensor(state).to(self._device).unsqueeze(0)
        if greedy:
            action = self.actor.policy.rsample(state)[2]
        else:
            action = self.actor.policy.rsample(state)[0]

        return action.detach().cpu().numpy()[0]

    def _get_q(
        self,
        state_batch: Float[torch.Tensor, "batch_size state_dim"],
        action_batch: Float[torch.Tensor, "batch_size action_dim"]
    ) -> Float[torch.Tensor, "batch_size"]:
        """
        Gets the Q values for `action_batch` actions in `state_batch` states
        from the critic, rather than the target critic.
        """
        if self._double_q:
            q1, q2 = self.critic.value_net(state_batch, action_batch)
            return torch.min(q1, q2)
        else:
            return self.critic.value_net(state_batch, action_batch)

    def update_critic(self) -> float:
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.buffer.sample(batch_size=self._batch_size)
        if state_batch is None:
            # Too few samples in the buffer to sample
            return

        if self._double_q:
            q_loss = self._update_double_critic(state_batch, action_batch, reward_batch,
                                                next_state_batch, mask_batch)

        else:
            q_loss = self._update_single_critic(state_batch, action_batch, reward_batch,
                                                next_state_batch, mask_batch)

        return q_loss

    def _update_single_critic(
        self,
        state_batch: Float[torch.Tensor, "batch_size state_dim"],
        action_batch: Float[torch.Tensor, "batch_size action_dim"],
        reward_batch: Float[torch.Tensor, "batch_size"],
        next_state_batch: Float[torch.Tensor, "batch_size state_dim"],
        mask_batch: Float[torch.Tensor, "batch_size"]
    ) -> float:
        """
        Update the critic using a batch of transitions when using a single Q
        critic.
        """
        if self._double_q:
            raise ValueError("cannot call _update_single_critic when using " +
                             "a double Q critic")

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            # Sample an action in the next state for the SARSA update
            next_state_action, next_state_log_pi = \
                self.actor.policy.sample(next_state_batch)[:2]

            if len(next_state_log_pi.shape) == 1:
                next_state_log_pi = next_state_log_pi.unsqueeze(-1)

            # Calculate the Q value of the next action in the next state
            q_next = self.critic.target_net(next_state_batch, next_state_action)

            if self._soft_q:
                q_next -= self._alpha * next_state_log_pi

            # Calculate the target for the SARSA update
            q_target = reward_batch + mask_batch * self._gamma * q_next

        # Calculate the Q value of each action in each respective state
        q = self.critic.value_net(state_batch, action_batch)

        # Calculate the loss between the target and estimate Q values
        q_loss = F.mse_loss(q, q_target)

        # Update the critic
        self.critic.optimizer.zero_grad()
        q_loss.backward()
        self.critic.optimizer.step()
        return q_loss.detach().item()

    def _update_double_critic(
        self,
        state_batch: Float[torch.Tensor, "batch_size state_dim"],
        action_batch: Float[torch.Tensor, "batch_size action_dim"],
        reward_batch: Float[torch.Tensor, "batch_size"],
        next_state_batch: Float[torch.Tensor, "batch_size state_dim"],
        mask_batch: Float[torch.Tensor, "batch_size"]
    ) -> float:
        """
        Update the critic using a batch of transitions when using a double Q
        critic.
        """
        if not self._double_q:
            raise ValueError("cannot call _update_single_critic when using " +
                             "a double Q critic")

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            # Sample an action in the next state for the SARSA update
            next_state_action, next_state_log_pi = \
                self.actor.policy.sample(next_state_batch)[:2]

            # Calculate the action values for the next state
            next_q1, next_q2 = self.critic.target_net(next_state_batch, next_state_action)

            # Double Q: target uses the minimum of the two computed action values
            min_next_q = torch.min(next_q1, next_q2)

            # If using soft action value functions, then adjust the target
            if self._soft_q:
                min_next_q -= self._alpha * next_state_log_pi

            # Calculate the target for the action value function update
            q_target = reward_batch + mask_batch * self._gamma * min_next_q

        # Calculate the two Q values of each action in each respective state
        q1, q2 = self.critic.value_net(state_batch, action_batch)

        # Calculate the losses on each critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q1_loss = F.mse_loss(q1, q_target)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = q1_loss + q2_loss

        # Update the critic
        self.critic.optimizer.zero_grad()
        q_loss.backward()
        self.critic.optimizer.step()
        return q_loss.detach().item()

    def update_actor(self) -> None:
        """
        Update the actor given a batch of transitions sampled from a replay
        buffer.
        """
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.buffer.sample(batch_size=self._batch_size)
        if state_batch is None:
            # Too few samples in the buffer to sample
            return
        # Calculate the actor loss
        if self._reparameterized:
            # Reparameterization trick
            if self._baseline_actions > 0:
                pi, log_pi = self.actor.policy.rsample(
                    state_batch,
                    num_samples=self._baseline_actions+1,
                )[:2]
                pi = pi.transpose(0, 1).reshape(
                    -1,
                    self._action_space.high.shape[0],
                )
                s_state_batch = state_batch.repeat_interleave(
                    self._baseline_actions + 1,
                    dim=0,
                )
                q = self._get_q(s_state_batch, pi)
                q = q.reshape(self._batch_size, self._baseline_actions + 1, -1)

                # Don't backprop through the approximate state-value baseline
                baseline = q[:, 1:].mean(axis=1).squeeze().detach()

                log_pi = log_pi[0, :, 0]
                q = q[:, 0, 0]
                q -= baseline
            else:
                pi, log_pi = self.actor.policy.rsample(state_batch)[:2]
                q = self._get_q(state_batch, pi)

            policy_loss = ((self._alpha * log_pi) - q).mean()

        else:
            # Log likelihood trick
            baseline = 0
            if self._baseline_actions > 0:
                with torch.no_grad():
                    pi = self.actor.policy.sample(
                        state_batch,
                        num_samples=self._baseline_actions,
                    )[0]
                    pi = pi.transpose(0, 1).reshape(
                        -1,
                        self._action_space.high.shape[0],
                    )
                    s_state_batch = state_batch.repeat_interleave(
                        self._baseline_actions,
                        dim=0,
                    )
                    q = self._get_q(s_state_batch, pi)
                    q = q.reshape(
                        self._batch_size,
                        self._baseline_actions,
                        -1,
                    )
                    baseline = q[:, 1:].mean(axis=1)

            sample = self.actor.policy.sample(
                state_batch,
                self._num_samples,
            )
            pi, log_pi = sample[:2]  # log_pi is differentiable

            if self._num_samples > 1:
                pi = pi.reshape(self._num_samples * self._batch_size, -1)
                state_batch = state_batch.repeat(self._num_samples, 1)

            with torch.no_grad():
                # Context manager ensures that we don't backprop through the q
                # function when minimizing the policy loss
                q = self._get_q(state_batch, pi)
                q -= baseline

            # Compute the policy loss
            log_pi = log_pi.reshape(self._num_samples * self._batch_size, -1)

            with torch.no_grad():
                scale = self._alpha * log_pi - q
            policy_loss = log_pi * scale
            policy_loss = policy_loss.mean()

        # Update the actor
        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()

        # Tune the entropy if appropriate
        if self._automatic_entropy_tuning:
            alpha_loss = -(self._log_alpha *
                           (log_pi + self._target_entropy).detach()).mean()

            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()

            self._alpha = self._log_alpha.exp().detach()

    def reset(self) -> None:
        pass
