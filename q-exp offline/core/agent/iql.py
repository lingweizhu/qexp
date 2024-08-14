import numpy as np
from core.agent import base
from core.network.network_architectures import FCNetwork
from core.policy.student import Student

import os
import torch
from torch.nn.utils import clip_grad_norm_

class IQL(base.ActorCritic):
    def __init__(self, cfg):
        super(IQL, self).__init__(cfg)

        self.clip_grad_param = 100
        self.temperature = 1./cfg.tau # inversed
        self.expectile = cfg.expectile #torch.FloatTensor([0.8]).to(device)

        self.value_net = FCNetwork(cfg.device, np.prod(cfg.state_dim), [cfg.hidden_units]*2, 1)
        self.value_optimizer = torch.optim.Adam(list(self.value_net.parameters()), cfg.q_lr)

    def compute_loss_pi(self, data):
        states, actions = data['obs'], data['act']
        with torch.no_grad():
            v = self.value_net(states)
        min_Q, _, _ = self.get_q_value_target(states, actions)
        exp_a = torch.exp((min_Q - v) * self.temperature)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(states.device))
        log_probs = self.ac.pi.log_prob(states, actions)
        actor_loss = -(exp_a * log_probs).mean()
        # print("pi", min_Q.size(), v.size(), exp_a.size(), log_probs.size())
        return actor_loss, log_probs
    
    def compute_loss_value(self, data):
        states, actions = data['obs'], data['act']
        min_Q, _, _ = self.get_q_value_target(states, actions)

        value = self.value_net(states)
        value_loss = self.expectile_loss(min_Q - value, self.expectile).mean()
        # print("value", min_Q.size(), value.size(), self.expectile_loss(min_Q - value, self.expectile).size())
        return value_loss

    def expectile_loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + (self.gamma * (1 - dones) * next_v)
        
        _, q1, q2 = self.get_q_value(states, actions, with_grad=True)
        critic1_loss = (0.5* (q_target - q1) ** 2).mean()
        critic2_loss = (0.5* (q_target - q2) ** 2).mean()
        loss_q = (critic1_loss + critic2_loss) * 0.5
        # print("q", q1.shape, q2.shape, q_target.shape, rewards.shape, dones.shape, actions.shape)
        return loss_q
        
    def update(self, data):
        self.value_optimizer.zero_grad()
        loss_vs = self.compute_loss_value(data)
        loss_vs.backward()
        self.value_optimizer.step()
        
        loss_q = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        clip_grad_norm_(self.ac.q1q2.parameters(), self.clip_grad_param)
        self.q_optimizer.step()

        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()

        loss_pi, _ = self.compute_loss_pi(data)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        return


    def save(self, timestamp=''):
        parameters_dir = self.parameters_dir
        params = {
            "actor_net": self.ac.pi.state_dict(),
            "critic_net": self.ac.q1q2.state_dict(),
            "value_net": self.value_net.state_dict()
        }
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        torch.save(params, path)
