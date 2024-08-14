import torch
import numpy as np
from gymnasium.envs.classic_control.continuous_mountain_car import (
    Continuous_MountainCarEnv
)

class Continuous_MountainCarEnvV2(Continuous_MountainCarEnv):
    """
    override gym's mountaincar to give reward of -1 per step
    and 100 on reaching the goal
    """
    def step(self, action: np.ndarray):
        state, reward, terminated, truncated, info = super().step(action)
        if terminated:
            reward = -1
        else:
            reward = -1
        return state, reward, terminated, False, {}


if __name__ == "__main__":
    import itertools
    class Agent:
        def decide(self, observation):
            position, velocity = observation
            if position > -4 * velocity or position < 13 * velocity - 0.6:
                force = 1.
            else:
                force = -1.
            action = np.array([force,])
            return action

    agent = Agent()
    env = Continuous_MountainCarEnvV2()
    observation = env.reset()[0]
    episode_reward = 0.
    for step in itertools.count():
        action = agent.decide(observation)
        observation, reward, t, tr, _ = env.step(action)
        episode_reward += reward
        if t or tr:
            print("finished episode")
            env.reset()
            episode_reward = 0
        if step == 200:
            break
        print('get {} rewards in {} steps'.format(episode_reward, step + 1))


