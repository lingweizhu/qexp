import os
import torch
from core.agent import base
from core.utils import torch_utils


class TD3BC(base.ActorCritic):
    def __init__(self, cfg):
        super(TD3BC, self).__init__(cfg)

        self.policy_freq = 1 #cfg.policy_freq
        self.alpha = cfg.tau

    def update(self, data):
        state, action, reward, next_state, not_done = data['obs'], data['act'], data['reward'], data['obs2'], 1.0 - data['done']
        action = torch_utils.tensor(action, self.device)
        with torch.no_grad():
            next_action, _ = self.ac.pi.sample(next_state)
            target_Q, _, _ = self.get_q_value_target(next_state, next_action)
            target_Q = reward + not_done * self.gamma * target_Q
        _, current_Q1, current_Q2 = self.get_q_value(state, action, with_grad=True)
        critic_loss = (torch.nn.functional.mse_loss(current_Q1, target_Q) +
                       torch.nn.functional.mse_loss(current_Q2, target_Q)) * 0.5
        # print("Q", next_action.size(), target_Q.size(), current_Q1.size(), current_Q2.size(), reward.size())

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        # Delayed policy updates
        if self.total_steps % self.policy_freq == 0:
            pi, _ = self.ac.pi.rsample(state)
            Q, _, _ = self.get_q_value(state, pi, with_grad=False)
            lmbda = self.alpha / Q.abs().mean().detach()
            actor_loss = -lmbda * Q.mean() + torch.nn.functional.mse_loss(pi, action)
            # print("pi", pi.size(), Q.size())
            # Optimize the actor
            self.pi_optimizer.zero_grad()
            actor_loss.backward()
            self.pi_optimizer.step()

        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()

    def save(self, timestamp=''):
        parameters_dir = self.parameters_dir
        params = {
            "actor_net": self.ac.pi.state_dict(),
            "critic_net": self.ac.q1q2.state_dict(),
        }
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        torch.save(params, path)

