import os

import numpy as np
import torch
import copy
from collections import namedtuple

from core.utils import torch_utils
from core.utils.buffer import Replay
from core.policy.gaussian import MLPDiscrete
from core.network.network_architectures import DoubleCriticNetwork, DoubleCriticDiscrete
from core.policy.factory import get_continuous_policy



class Base:
    def __init__(self, cfg):
        self.exp_path = cfg.exp_path
        self.seed = cfg.seed
        self.use_target_network = cfg.use_target_network
        self.target_network_update_freq = cfg.target_network_update_freq
        self.parameters_dir = self.get_parameters_dir()

        self.batch_size = cfg.batch_size
        self.env = cfg.env_fn()
        self.eval_env = copy.deepcopy(cfg.env_fn)()
        self.offline_data = cfg.offline_data
        self.replay = Replay(memory_size=2000000, batch_size=cfg.batch_size, seed=cfg.seed)
        self.state_normalizer = lambda x: x
        self.evaluation_criteria = cfg.evaluation_criteria
        self.logger = cfg.logger
        self.timeout = cfg.timeout
        self.action_dim = cfg.action_dim

        self.gamma = cfg.gamma
        self.device = 'cpu'
        self.stats_queue_size = 5
        self.episode_reward = 0
        self.episode_rewards = []
        self.total_steps = 0
        self.reset = True
        self.ep_steps = 0
        self.num_episodes = 0
        self.ep_returns_queue_train = np.zeros(self.stats_queue_size)
        self.ep_returns_queue_test = np.zeros(self.stats_queue_size)
        self.train_stats_counter = 0
        self.test_stats_counter = 0
        self.agent_rng = np.random.RandomState(self.seed)

        self.populate_latest = False
        self.populate_states, self.populate_actions, self.populate_true_qs = None, None, None
        self.automatic_tmp_tuning = False
        
        self.state = None
        self.action = None
        self.next_state = None
        self.eps = 1e-8
        self.cfg = cfg

    def get_parameters_dir(self):
        d = os.path.join(self.exp_path, "parameters")
        torch_utils.ensure_dir(d)
        return d

    def offline_param_init(self):
        self.trainset = self.training_set_construction(self.offline_data)
        self.training_size = len(self.trainset[0])
        self.training_indexs = np.arange(self.training_size)

        self.training_loss = []
        self.test_loss = []
        self.tloss_increase = 0
        self.tloss_rec = np.inf

    def get_data(self):
        states, actions, rewards, next_states, terminals = self.replay.sample()
        in_ = torch_utils.tensor(self.state_normalizer(states), self.device)
        actions = torch_utils.tensor(actions, self.device)
        r = torch_utils.tensor(rewards, self.device).view((-1, 1))
        ns = torch_utils.tensor(self.state_normalizer(next_states), self.device)
        t = torch_utils.tensor(terminals, self.device).view((-1, 1))
        data = {
            'obs': in_,
            'act': actions,
            'reward': r,
            'obs2': ns,
            'done': t
        }
        return data

    def fill_offline_data_to_buffer(self):
        self.trainset = self.training_set_construction(self.offline_data)
        train_s, train_a, train_r, train_ns, train_t = self.trainset
        for idx in range(len(train_s)):
            self.replay.feed([train_s[idx], train_a[idx], train_r[idx], train_ns[idx], train_t[idx]])

    def step(self):
        # trans = self.feed_data()
        self.update_stats(0, None)
        data = self.get_data()
        losses = self.update(data)
        return losses
    
    def update(self, data):
        raise NotImplementedError
        
    def update_stats(self, reward, done):
        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        if done or self.ep_steps == self.timeout:
            self.episode_rewards.append(self.episode_reward)
            self.num_episodes += 1
            if self.evaluation_criteria == "return":
                self.add_train_log(self.episode_reward)
            elif self.evaluation_criteria == "steps":
                self.add_train_log(self.ep_steps)
            else:
                raise NotImplementedError
            self.episode_reward = 0
            self.ep_steps = 0
            self.reset = True

    def add_train_log(self, ep_return):
        self.ep_returns_queue_train[self.train_stats_counter] = ep_return
        self.train_stats_counter += 1
        self.train_stats_counter = self.train_stats_counter % self.stats_queue_size

    def add_test_log(self, ep_return):
        self.ep_returns_queue_test[self.test_stats_counter] = ep_return
        self.test_stats_counter += 1
        self.test_stats_counter = self.test_stats_counter % self.stats_queue_size

    def populate_returns(self, log_traj=False, total_ep=None, initialize=False):
        total_ep = self.stats_queue_size if total_ep is None else total_ep
        total_steps = 0
        total_states = []
        total_actions = []
        total_returns = []
        for ep in range(total_ep):
            ep_return, steps, traj = self.eval_episode(log_traj=log_traj)
            total_steps += steps
            total_states += traj[0]
            total_actions += traj[1]
            total_returns += traj[2]
            if self.evaluation_criteria == "return":
                self.add_test_log(ep_return)
                if initialize:
                    self.add_train_log(ep_return)
            elif self.evaluation_criteria == "steps":
                self.add_test_log(steps)
                if initialize:
                    self.add_train_log(steps)
            else:
                raise NotImplementedError
        return [total_states, total_actions, total_returns]

    def eval_episode(self, log_traj=False):
        ep_traj = []
        state = self.eval_env.reset()
        total_rewards = 0
        ep_steps = 0
        done = False
        while True:
            action = self.eval_step(state)
            last_state = state
            state, reward, done, _ = self.eval_env.step(action)
            if log_traj:
                ep_traj.append([last_state, action[0], reward])
            total_rewards += reward
            ep_steps += 1
            if done or ep_steps == self.timeout:
                break

        states = []
        actions = []
        rets = []
        if log_traj:
            ret = 0
            for i in range(len(ep_traj)-1, -1, -1):
                s, a, r = ep_traj[i]
                ret = r + self.gamma * ret
                rets.insert(0, ret)
                actions.insert(0, a)
                states.insert(0, s)
        return total_rewards, ep_steps, [states, actions, rets]

    def eval_step(self, o):
        raise NotImplementedError

    def log_return(self, log_ary, name, elapsed_time):
        rewards = log_ary
        total_episodes = len(self.episode_rewards)
        mean, median, min_, max_ = np.mean(rewards), np.median(rewards), np.min(rewards), np.max(rewards)

        log_str = '%s LOG: steps %d, episodes %3d, ' \
                  'returns %.2f/%.2f/%.2f/%.2f/%d (mean/median/min/max/num), %.2f steps/s'

        self.logger.info(log_str % (name, self.total_steps, total_episodes, mean, median,
                                        min_, max_, len(rewards),
                                        elapsed_time))
        return mean, median, min_, max_

    def log_file(self, elapsed_time=-1, test=True):
        mean, median, min_, max_ = self.log_return(self.ep_returns_queue_train, "TRAIN", elapsed_time)
        if test:
            self.populate_states, self.populate_actions, self.populate_true_qs = self.populate_returns(log_traj=True)
            self.populate_latest = True
            mean, median, min_, max_ = self.log_return(self.ep_returns_queue_test, "TEST", elapsed_time)
            try:
                normalized = np.array([self.eval_env.env.unwrapped.get_normalized_score(ret_) for ret_ in self.ep_returns_queue_test])
                mean, median, min_, max_ = self.log_return(normalized, "Normalized", elapsed_time)
            except:
                pass
        return mean, median, min_, max_

    def training_set_construction(self, data_dict):
        assert len(list(data_dict.keys())) == 1
        data_dict = data_dict[list(data_dict.keys())[0]]
        states = data_dict['states']
        actions = data_dict['actions']
        rewards = data_dict['rewards']
        next_states = data_dict['next_states']
        terminations = data_dict['terminations']
        return [states, actions, rewards, next_states, terminations]


class ActorCritic(Base):
    def __init__(self, cfg):
        super(ActorCritic, self).__init__(cfg)

        pi = self.get_policy_func(cfg.discrete_control, cfg)
        q1q2 = self.get_critic_func(cfg.discrete_control, cfg.device, cfg.state_dim, cfg.action_dim, cfg.hidden_units)
        AC = namedtuple('AC', ['q1q2', 'pi'])
        self.ac = AC(q1q2=q1q2, pi=pi)
        pi_target = self.get_policy_func(cfg.discrete_control, cfg)
        q1q2_target = self.get_critic_func(cfg.discrete_control, cfg.device, cfg.state_dim, cfg.action_dim, cfg.hidden_units)
        q1q2_target.load_state_dict(q1q2.state_dict())
        pi_target.load_state_dict(pi.state_dict())
        ACTarg = namedtuple('ACTarg', ['q1q2', 'pi'])
        self.ac_targ = ACTarg(q1q2=q1q2_target, pi=pi_target)
        self.ac_targ.q1q2.load_state_dict(self.ac.q1q2.state_dict())
        self.ac_targ.pi.load_state_dict(self.ac.pi.state_dict())
        self.pi_optimizer = torch.optim.Adam(list(self.ac.pi.parameters()), cfg.pi_lr)
        self.q_optimizer = torch.optim.Adam(list(self.ac.q1q2.parameters()), cfg.q_lr)

        if cfg.discrete_control:
            self.get_q_value = self.get_q_value_discrete
            self.get_q_value_target = self.get_q_value_target_discrete
        else:
            self.get_q_value = self.get_q_value_cont
            self.get_q_value_target = self.get_q_value_target_cont

        self.polyak = cfg.polyak
        self.fill_offline_data_to_buffer()
        self.offline_param_init()

    def get_policy_func(self, discrete_control, cfg):
        if discrete_control:
            pi = MLPDiscrete(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units] * 2)
        else:
            pi = get_continuous_policy(cfg.distribution, cfg)
        return pi

    def get_critic_func(self, discrete_control, device, state_dim, action_dim, hidden_units):
        if discrete_control:
            q1q2 = DoubleCriticDiscrete(device, state_dim, [hidden_units] * 2, action_dim)
        else:
            q1q2 = DoubleCriticNetwork(device, state_dim, action_dim, [hidden_units] * 2)
        return q1q2

    def eval_step(self, o):
        o = torch_utils.tensor(self.state_normalizer(o).reshape((1, -1)), self.device)
        with torch.no_grad():
            a, _ = self.ac.pi.sample(o, deterministic=False)
        a = torch_utils.to_np(a)
        return a

    def get_q_value_discrete(self, o, a, with_grad=False):
        if with_grad:
            q1_pi, q2_pi = self.ac.q1q2(o)
            q1_pi, q2_pi = q1_pi[np.arange(len(a)), a], q2_pi[np.arange(len(a)), a]
            q_pi = torch.min(q1_pi, q2_pi)
        else:
            with torch.no_grad():
                q1_pi, q2_pi = self.ac.q1q2(o)
                q1_pi, q2_pi = q1_pi[np.arange(len(a)), a], q2_pi[np.arange(len(a)), a]
                q_pi = torch.min(q1_pi, q2_pi)
        return q_pi, q1_pi, q2_pi

    def get_q_value_target_discrete(self, o, a):
        with torch.no_grad():
            q1_pi, q2_pi = self.ac_targ.q1q2(o)
            q1_pi, q2_pi = q1_pi[np.arange(len(a)), a], q2_pi[np.arange(len(a)), a]
            q_pi = torch.min(q1_pi, q2_pi)
        return q_pi, q1_pi, q2_pi

    def get_q_value_cont(self, o, a, with_grad=False):
        if with_grad:
            q1_pi, q2_pi = self.ac.q1q2(o, a)
            q_pi = torch.min(q1_pi, q2_pi)
        else:
            with torch.no_grad():
                q1_pi, q2_pi = self.ac.q1q2(o, a)
                q_pi = torch.min(q1_pi, q2_pi)
        return q_pi, q1_pi, q2_pi

    def get_q_value_target_cont(self, o, a):
        with torch.no_grad():
            q1_pi, q2_pi = self.ac_targ.q1q2(o, a)
            q_pi = torch.min(q1_pi, q2_pi)
        return q_pi, q1_pi, q2_pi

    def sync_target(self):
        with torch.no_grad():
            for p, p_targ in zip(self.ac.q1q2.parameters(), self.ac_targ.q1q2.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.ac.pi.parameters(), self.ac_targ.pi.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
