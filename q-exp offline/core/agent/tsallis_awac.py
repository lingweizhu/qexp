import numpy as np
from core.agent import base
import os
import torch

from core.network.network_architectures import FCNetwork
from core.policy.student import Student


class TKLPolicyInAC(base.ActorCritic):
    def __init__(self, cfg):
        super(TKLPolicyInAC, self).__init__(cfg)

        self.value_net = FCNetwork(cfg.device, np.prod(cfg.state_dim), [cfg.hidden_units]*2, 1)

        self.value_optimizer = torch.optim.Adam(list(self.value_net.parameters()), cfg.q_lr)
        # self.beh_pi_optimizer = torch.optim.Adam(
        #     list(self.beh_pi.parameters()), cfg.pi_lr)

        self.tau = cfg.tau

        self.q = cfg.tsallis_q
        self.exp_threshold = 10000

    def logq_x(self, x):
        return (x**(self.q-1) - 1) / (self.q-1)
    
    def expq_x(self, x):
        return torch.pow(torch.clip(1 + (self.q-1)*x, min=0), 1/(self.q-1))
        

    # def compute_loss_beh_pi(self, data):
    #     """L_{\omega}, learn behavior policy"""
    #     states, actions = data['obs'], data['act']
    #     beh_log_probs = self.beh_pi.log_prob(states, actions)
    #     beh_loss = -beh_log_probs.mean()
    #     return beh_loss, beh_log_probs

    def compute_loss_value(self, data):
        """L_{\phi}, learn z for state value, v = tau log z"""
        states = data['obs']
        v_phi = self.value_net(states)
        with torch.no_grad():
            actions, log_probs = self.ac.pi.sample(states)
            min_Q, _, _ = self.get_q_value_target(states, actions)
        target = min_Q 
        value_loss = (0.5 * (v_phi - target) ** 2).mean()
        # print("v", v_phi.size(), target.size(), min_Q.size(), actions.size(), log_probs.size())
        return value_loss, v_phi.detach().numpy(), log_probs.detach().numpy()
    
    def get_state_value(self, state):
        with torch.no_grad():
            value = self.value_net(state)
        return value

    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        with torch.no_grad():
            next_actions, _ = self.ac.pi.sample(next_states)
        next_q_values, _, _ = self.get_q_value_target(next_states, next_actions)
        q_target = rewards + self.gamma * (1 - dones) * next_q_values
        
        minq, q1, q2 = self.get_q_value(states, actions, with_grad=True)    
        critic1_loss = (0.5 * (q_target - q1) ** 2).mean()
        critic2_loss = (0.5 * (q_target - q2) ** 2).mean()
        loss_q = (critic1_loss + critic2_loss) * 0.5
        q_info = minq.detach().numpy()
        # print("q", minq.size(), q1.size(), q2.size(), next_q_values.size(), rewards.size(), dones.size())
        return loss_q, q_info

    def compute_loss_pi(self, data):
        """L_{\psi}, extract learned policy"""
        states, actions = data['obs'], data['act']
        log_probs = self.ac.pi.log_prob(states, actions)
        min_Q, _, _ = self.get_q_value(states, actions, with_grad=False)

        with torch.no_grad():
            value = self.get_state_value(states)
        x = (min_Q - value) / self.tau
                  
        tsallis_policy = self.expq_x(x)
        clipped = torch.clip(tsallis_policy, self.eps, self.exp_threshold)
        pi_loss = -(clipped * log_probs).mean()
        # print("pi", clipped.size(), log_probs.size(), tsallis_policy.size(), value.size(), min_Q.size(), psi.size())
        return pi_loss, ""

    # def update_beta(self, data):
    #     loss_beh_pi, _ = self.compute_loss_beh_pi(data)
    #     self.beh_pi_optimizer.zero_grad()
    #     loss_beh_pi.backward()
    #     self.beh_pi_optimizer.step()
    #     return loss_beh_pi

    def update(self, data):
        # loss_beta = self.update_beta(data).item()
        
        self.value_optimizer.zero_grad()
        loss_vs, v_info, logp_info = self.compute_loss_value(data)
        loss_vs.backward()
        self.value_optimizer.step()

        loss_q, qinfo = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        loss_pi, _ = self.compute_loss_pi(data)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
        
        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()

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

    def load(self, parameters_dir, timestamp=''):
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        model = torch.load(path)
        self.ac.pi.load_state_dict(model["actor_net"])
        self.ac.q1q2.load_state_dict(model["critic_net"])
        self.value_net.load_state_dict(model["value_net"])
