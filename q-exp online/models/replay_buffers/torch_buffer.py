import torch
from .experience_replay import ExperienceReplay


class TorchBuffer(ExperienceReplay):
    """
    Class TorchBuffer implements an experience replay buffer. The
    difference between this class and the ExperienceReplay class is that this
    class keeps all experiences as a torch Tensor on the appropriate device
    so that if using PyTorch, we do not need to cast the batch to a
    FloatTensor every time we sample and then place it on the appropriate
    device, as this is very time consuming. This class is basically a
    PyTorch efficient implementation of ExperienceReplay.
    """
    def __init__(self, capacity: int, seed: int, state_size: int,
                 action_size: int, device: torch.device) -> None:
        """
        Constructor

        Parameters
        ----------
        capacity : int
            The capacity of the buffer
        seed : int
            The random seed used for sampling from the buffer
        device : torch.device
            The device on which the buffer instances should be stored
        state_size : int
            The number of dimensions in the state feature vector
        action_size : int
            The number of dimensions in the action vector
        """
        super().__init__(capacity, seed, state_size, action_size, device)
        self.cast = torch.from_numpy

    def init_buffer(self) -> int:
        self.state_buffer = torch.FloatTensor(self.capacity, *self.state_size)
        self.state_buffer = self.state_buffer.to(self.device)

        self.next_state_buffer = torch.FloatTensor(self.capacity,
                                                   *self.state_size)
        self.next_state_buffer = self.next_state_buffer.to(self.device)

        self.action_buffer = torch.FloatTensor(self.capacity, *self.action_size)
        self.action_buffer = self.action_buffer.to(self.device)

        self.reward_buffer = torch.FloatTensor(self.capacity, 1)
        self.reward_buffer = self.reward_buffer.to(self.device)

        self.done_buffer = torch.FloatTensor(self.capacity, 1)
        self.done_buffer = self.done_buffer.to(self.device)

