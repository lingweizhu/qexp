import torch
import torch.nn as nn
from gymnasium import Env
from jaxtyping import Float
from .optimizer_factory import get_optimizer
from .policy_parameterizations.factory import get_policy
from .critic import get_critic
import copy


class Actor:
    def __init__(self, cfg: dict, env: Env, store_old_policy: bool = False, device: str = 'cpu') -> None:
        """
        cfg contains:
            - policy parameters
            - optimizer params
        env: Environment to infer action and state shapes from
        store_old_policy: see self.policy_backup_hook() for details
        """
        self.policy = get_policy(cfg.policy, env, device)
        self.policy_old = None
        self.optimizer = get_optimizer(cfg.optimizer, list(self.policy.parameters()))
        if store_old_policy:
            self.policy_old = copy.deepcopy(self.policy)
            self.hook = self.optimizer.register_step_pre_hook(self.policy_backup_hook)

    def policy_backup_hook(self, *args) -> None:
        """
        store the policy's parameters in self.policy_old whenever we call
        a self.optimzier.step(). This is so we will always have a set of
        previous policy parameters available for use in KL computations
        """
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.zero_grad()

    def sample(self, state: Float[torch.Tensor, "state_dim"]) -> Float[torch.Tensor, "action_dim"]:
        return self.policy.sample(state)
    

class PPOActor:
    """
    PPO actor contains critic and optimizes them using one loss

    NOTE: cfg here is cfg.agent, it is not cfg.agent.actor as in the Actor class
    """
    def __init__(self, cfg: dict, env: Env, store_old_policy: bool = False, device: str = 'cpu') -> None:
        
        self.policy = get_policy(cfg.actor.policy, env, device)
        self.critic = get_critic(cfg.critic, env, device)

        self.policy_old = None
        self.optimizer = get_optimizer(cfg.actor.optimizer, \
                    list(self.policy.parameters()) + list(self.critic.value_net.parameters()))
        if store_old_policy:
            self.policy_old = copy.deepcopy(self.policy)
            self.hook = self.optimizer.register_step_pre_hook(self.policy_backup_hook)

    def policy_backup_hook(self, *args) -> None:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.zero_grad()

    def sample(self, state: Float[torch.Tensor, "state_dim"]) -> Float[torch.Tensor, "action_dim"]:
        return self.policy.sample(state)

if __name__ == "__main__":
    pass

