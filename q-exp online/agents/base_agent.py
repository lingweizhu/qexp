from abc import ABC, abstractmethod
from jaxtyping import Float
import numpy as np

class BaseAgent(ABC):
    """
    Class BaseAgent implements the base functionality for all agents
    """
    def __init__(self):
        pass

    @abstractmethod
    def act(self, state: Float[np.ndarray, "state_dim"]) -> Float[np.ndarray, "action_dim"]:
        pass

    @abstractmethod
    def update_critic(self) -> float:
        pass

    @abstractmethod
    def update_actor(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class OnPolicyAgent(BaseAgent):
    """
    Class BaseAgent implements the base functionality for all agents
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_action_and_value(self) -> Float[np.ndarray, '...']:
        pass
