import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from .mountaincar_continuous_v2 import Continuous_MountainCarEnvV2
from .acrobot import AcrobotEnv


def get_env(cfg: dict, seed: int):
    gym.envs.register(
        id='MountainCarContinuous-v2',
        entry_point=Continuous_MountainCarEnvV2,
        max_episode_steps=cfg.max_episode_steps
    )

    if cfg.name == "AcrobotContinuous":
        env = AcrobotEnv(seed=seed, continuous_action=True)
        env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
        return env
    elif cfg.gym:
        return gym.make(cfg.name, max_episode_steps=cfg.max_episode_steps)
    else:
        raise NotImplementedError
