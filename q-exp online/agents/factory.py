import gymnasium
from copy import deepcopy
from typing import Optional
from gymnasium.spaces.utils import flatdim
from .base_agent import BaseAgent
from .greedyac import GreedyAC
from .ppo import PPO
from .mpo import MPO
from .td3 import TD3
from .sac import SAC
from .tsallis_awac import TsallisAWAC


def get_agent(cfg: dict,
              discrete_action: bool,
              device: str,
              env: object,
              actor: object,
              critic: object,
              replay_buffer: object,

              ) -> BaseAgent:
    if isinstance(env, gymnasium.Env):
        state_dim = flatdim(env.observation_space)
        action_dim = flatdim(env.action_space)
        action_space = env.action_space
    else:
        raise NotImplementedError
    match cfg.name:
        case "greedyac":
            proposal_actor = deepcopy(actor)
            return GreedyAC(discrete_action=discrete_action,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            gamma=cfg.gamma,
                            batch_size=cfg.buffer.batch_size,
                            alpha=cfg.alpha,
                            device=device,
                            behavior_policy=actor,
                            proposal_policy=proposal_actor,
                            critic=critic,
                            replay_buffer=replay_buffer,
                            rho=cfg.rho,
                            n_action_proposals=cfg.n_action_proposals,
                            entropy_from_single_sample=cfg.entropy_from_single_sample)

        case "mpo":
            return MPO(discrete_action=discrete_action,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            gamma=cfg.gamma,
                            batch_size=cfg.buffer.batch_size,
                            alpha=cfg.alpha,
                            device=device,
                            actor=actor,
                            critic=critic,
                            replay_buffer=replay_buffer,
                            rho=cfg.rho,
                            n_action_proposals=cfg.n_action_proposals,
                            )
        case "td3":
            return TD3(discrete_action=discrete_action,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            gamma=cfg.gamma,
                            batch_size=cfg.buffer.batch_size,
                            device=device,
                            actor=actor,
                            critic=critic,
                            replay_buffer=replay_buffer,
                            action_space=action_space,
                            exploration_noise=cfg.exploration_noise,
                            policy_noise=cfg.policy_noise,
                            noise_clip=cfg.noise_clip,
                            )
        case "sac":
            double_q = False
            if cfg.critic.network.name == "double_q_net":
                double_q = True
            return SAC(action_space=action_space,
                       gamma = cfg.gamma,
                       batch_size=cfg.buffer.batch_size,
                       alpha=cfg.alpha,
                       alpha_lr=cfg.alpha_lr,
                       device=device,
                       actor=actor,
                       critic=critic,
                       replay_buffer=replay_buffer,
                       baseline_actions=cfg.baseline_actions,
                       reparameterized=cfg.reparameterized,
                       soft_q=cfg.soft_q,
                       double_q=double_q,
                       n_samples_for_entropy=cfg.n_samples_for_entropy,
                       automatic_entropy_tuning=cfg.automatic_entropy_tuning)
        case "tsallis_awac":
            return TsallisAWAC(discrete_action=discrete_action,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            gamma=cfg.gamma,
                            batch_size=cfg.buffer.batch_size,
                            alpha=cfg.alpha,
                            device=device,
                            actor=actor,
                            critic=critic,
                            replay_buffer=replay_buffer,
                            rho=cfg.rho,
                            n_action_proposals=cfg.n_action_proposals,
                            entropic_index=cfg.entropic_index,
                            )        
        case _:
            raise NotImplementedError

