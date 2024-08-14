import torch
import gymnasium
from gymnasium.spaces.utils import flatdim
from .experience_replay import ExperienceReplay
from .torch_buffer import TorchBuffer
from .rollout_buffer import TorchRolloutBuffer

def get_buffer(cfg: dict,
               seed: int,
               env: object,
               device: torch.device) -> ExperienceReplay:
    if isinstance(env, gymnasium.Env):
        state_dim = env.observation_space.shape
        action_dim = env.action_space.shape
        action_space = env.action_space
    else:
        raise NotImplementedError

    match cfg.type:
        case "basic":
            return ExperienceReplay(capacity=cfg.capacity,
                                    seed=seed,
                                    state_size=state_dim,
                                    action_size=action_dim,
                                    device=device)
        case "torchbuffer":
            return TorchBuffer(capacity=cfg.capacity,
                               seed=seed,
                               state_size=state_dim,
                               action_size=action_dim,
                               device=device)
        case _:
            raise NotImplementedError
