from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .base_agent import OnPolicyAgent
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from jaxtyping import Float
from copy import deepcopy
import sys
sys.path.append("../utils")


class PPO(OnPolicyAgent):
    def __init__(self, 
                 discrete_action: bool,
                 action_dim: int, 
                 state_dim: int, 
                 gamma: float,
                 actor: torch.nn.Module,
                 rollout_buffer: torch.nn.Module,
                 episode_steps: int,
                 minibatches: int,
                 training_epochs: int, 
                 eps_clip: float,  
                 vloss_coef: float,
                 entropy_coef: float,
                 clip_vloss: bool,
                 device=torch.device,
                 ):
        super().__init__()        
        self.device = device
        self.has_continuous_action_space = ~discrete_action

        self.gamma = gamma
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.eps_clip = eps_clip
        self.training_epochs = training_epochs
        
        self.buffer = rollout_buffer
        self.actor = actor

        self.vloss_coef = vloss_coef
        self.entropy_coef = entropy_coef

        self.batch_size = episode_steps
        self.minibatches = minibatches
        self.minibatch_size = int(self.batch_size // minibatches)

        self.clip_vloss = clip_vloss

    @torch.no_grad()
    def act(self, state: Float[np.ndarray, "state_dim"], greedy: bool = False) -> Float[np.ndarray, "action_dim"]:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, action_mean = self.actor.sample(state)
        act = action.detach().cpu().numpy()[0]
        if greedy:
            act = action_mean.detach().cpu().numpy()[0]
        if self.has_continuous_action_space:
            return act
        else:
            return int(act[0])



    def get_action_and_value(self, state, action=None):
        """
        on policy agents evaluate state value and log-probs etc
        when encountering these transitions
        """
        with torch.no_grad():
            state = torch.FloatTensor(state)
            if action is None:
                action = self.act(state)          
            action = torch.FloatTensor(action)                
            logprobs = self.actor.policy.log_prob(state, action).numpy()
            entropy = self.actor.policy.entropy(state).numpy()
            value = self.actor.critic.value_net(state).numpy()

        return action.numpy(), logprobs, entropy, value



    def update_critic(self) -> float:
        pass
    

    def update_actor(self, log) -> None:

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, value_batch, logprob_batch = self.buffer.get_sample()
        if state_batch is None:
            # Too few samples in the buffer to sample
            print("it should never print anything")
            return

        returns = self.compute_returns(reward_batch, mask_batch)
        batch_idx = np.arange(self.batch_size)
        for epoch in range(self.training_epochs):
            np.random.shuffle(batch_idx)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_idx = batch_idx[start:end]
                # minibatch_idx = batch_idx

                newlogprob = self.actor.policy.log_prob(state_batch[minibatch_idx], action_batch[minibatch_idx])
                entropy = self.actor.policy.entropy(state_batch[minibatch_idx])
                newvalue = self.actor.critic.value_net(state_batch[minibatch_idx])
                logratio = newlogprob - logprob_batch[minibatch_idx]
                ratio = logratio.exp()
            
                minibatch_advantages = returns[minibatch_idx].detach().unsqueeze(-1) - value_batch[minibatch_idx]
                minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -minibatch_advantages * ratio
                pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                entropy_loss = entropy.mean()

                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - returns[minibatch_idx]) ** 2
                    v_clipped = returns[minibatch_idx] + torch.clamp(newvalue - value_batch[minibatch_idx], -self.eps_clip, self.eps_clip)
                    v_loss_clipped = (v_clipped - returns[minibatch_idx]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = F.mse_loss(newvalue, returns[minibatch_idx].detach())
                
                loss = pg_loss - self.entropy_coef * entropy_loss + self.vloss_coef * v_loss

                self.actor.optimizer.zero_grad()
                loss.backward()
                self.actor.optimizer.step()

        #     log.info(f'epoch {epoch},\
        #             pg loss {pg_loss.mean().detach().numpy().item():.2f}, \
        #             entropy: {entropy_loss.detach().numpy().item():.2f} \t \
        #             v loss: {v_loss.detach().numpy().item():.2f}'
        #     )            

    def reset(self) -> None:
        pass    



    def compute_returns(self, reward_batch, mask_batch):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(reward_batch), reversed(mask_batch)):
            if not is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        return rewards
       
