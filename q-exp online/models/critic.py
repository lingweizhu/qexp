import torch
import torch.nn as nn
from jaxtyping import Float
from gymnasium import Env
from .optimizer_factory import get_optimizer
from .value_networks.factory import get_value_network
import sys
sys.path.append("../../utils")
from utils.nn_utils import soft_update, hard_update


def get_critic(cfg: dict, env: Env, device: str) -> object:
    
    if hasattr(cfg.network, "use_td3_critic"):
        return TD3Critic(cfg, env, device)
    elif cfg.network.use_target_network:
        return DoubleCritic(cfg, env, device)
    else:
        return Critic(cfg, env, device)


class Critic:
    def __init__(self, cfg: dict, env: object, device: str) -> None:
        """
        cfg contains:
            - optimizer params
            - value network params
        Env: environment to fetch action/state dims from
        """
        self.value_net = get_value_network(cfg.network, env).to(device)
        self.optimizer = get_optimizer(cfg.optimizer, list(self.value_net.parameters()))

    def get_qvalues(self,
                    states: Float[torch.Tensor, "batch state_dim"],
                    actions: Float[torch.Tensor, "batch action_dim"]) -> Float[torch.Tensor, "batch"]:
        return self.value_net(torch.FloatTensor(states), torch.FloatTensor(actions))


class DoubleCritic:
    def __init__(self, cfg: dict, env: object, device: str) -> None:
        """
        This critic uses target network
        cfg contains:
            - optimizer params
            - value network params
        """
        self.value_net = get_value_network(cfg.network, env).to(device)
        self.target_net= get_value_network(cfg.network, env).to(device)
        self.optimizer = get_optimizer(cfg.optimizer, list(self.value_net.parameters()))
        # make the target_net init same as value_net
        hard_update(self.target_net, self.value_net)
        # target net cfg
        self.tau = cfg.network.tau
        self.target_update_interval = cfg.network.target_update_interval
        self.updates_done = 0
        # automatically trigger the target net update hook whenever grad step is done
        self.hook = self.optimizer.register_step_post_hook(self.update_target_net)

    def update_target_net(self, *args) -> None:
        """
        use this as a hook to automatically update the target network
        parameters
        """
        self.updates_done += 1
        if self.updates_done % self.target_update_interval == 0:
            soft_update(self.target_net, self.value_net, self.tau)


class TD3Critic:
    def __init__(self, cfg: dict, env: object, device: str) -> None:
        """
        This critic uses target network
        cfg contains:
            - optimizer params
            - value network params
        """
        self.value_net_1 = get_value_network(cfg.network, env).to(device)
        self.value_net_2 = get_value_network(cfg.network, env).to(device)
        self.target_net_1 = get_value_network(cfg.network, env).to(device)
        self.target_net_2 = get_value_network(cfg.network, env).to(device)
        self.optimizer = get_optimizer(cfg.optimizer, \
                list(self.value_net_1.parameters()) + list(self.value_net_2.parameters()))
        # make the target_net init same as value_net
        hard_update(self.target_net_1, self.value_net_1)
        hard_update(self.target_net_2, self.value_net_2)
        # target net cfg
        self.tau = cfg.network.tau
        self.target_update_interval = cfg.network.target_update_interval
        self.updates_done = 0
        # automatically trigger the target net update hook whenever grad step is done
        self.hook = self.optimizer.register_step_post_hook(self.update_target_net)

    def update_target_net(self, *args) -> None:
        """
        use this as a hook to automatically update the target network
        parameters
        """
        self.updates_done += 1
        if self.updates_done % self.target_update_interval == 0:
            soft_update(self.target_net_1, self.value_net_1, self.tau)        
            soft_update(self.target_net_2, self.value_net_2, self.tau)        


if __name__ == "__main__":
    pass

