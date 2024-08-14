import numpy as np
import torch
from abc import ABC, abstractmethod
from jaxtyping import Float


class RolloutBuffer(ABC):
    """
    Abstract base class ExperienceReplay implements an experience replay
    buffer. The specific kind of buffer is determined by classes which
    implement this base class. For example, NumpyBuffer stores all
    transitions in a numpy array while TorchBuffer implements the buffer
    as a torch tensor.

    Attributes
    ----------
    self.cast : func
        A function which will cast data into an appropriate form to be
        stored in the replay buffer. All incoming data is assumed to be
        a numpy array.
    """
    def __init__(self, capacity: int, seed: int, state_size: int,
                 action_size: int, device: torch.device=None) -> None:
        
        self.device = device
        self.is_full = False
        self.position = 0
        self.capacity = capacity
   

        # Set the casting function, which is needed for implementations which
        # may keep the ER buffer as a different data structure, for example
        # a torch tensor, in this case all data needs to be cast to a torch
        # tensor before storing
        self.cast = lambda x: x

        # Set the random number generator
        self.random = np.random.default_rng(seed=seed)

        # Save the size of states and actions
        self.state_size = state_size
        self.action_size = action_size

        self._sampleable = False

        # Buffer of state, action, reward, next_state, done
        self.state_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.next_state_buffer = None
        self.done_buffer = None
        self.value_buffer = None
        self.logprob_buffer = None
        
        self.init_buffer()

    @abstractmethod
    def init_buffer(self) -> None:
        """
        Initializes the buffers on which to store transitions.

        Note that different classes which implement this abstract base class
        may use different data types as buffers. For example, NumpyBuffer
        stores all transitions using a numpy array, while TorchBuffer
        stores all transitions on a torch Tensor on a specific device in order
        to speed up training by keeping transitions on the same device as
        the device which holds the model.

        Post-Condition
        --------------
        The replay buffer self.buffer has been initialized
        """
        pass

    def push(self, 
            state: Float[np.ndarray, "{self.state_size}"],
            action: Float[np.ndarray, "{self.action_size}"],
            reward: float,
            next_state: Float[np.ndarray, "{self.state_size}"],
            done: bool,
            value: Float[np.ndarray, "{self.action_size}"],
            logprob: Float[np.ndarray, "{self.action_size}"],
            ) -> None:

        reward = np.array([reward])
        done = np.array([done])

        state = self.cast(state)
        action = self.cast(action)
        reward = self.cast(reward)
        next_state = self.cast(next_state)
        done = self.cast(done)
        value = self.cast(value)
        logprob = self.cast(logprob)

        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = done
        self.value_buffer[self.position] = value
        self.logprob_buffer[self.position] = logprob

        if self.position >= self.capacity - 1:
            self.is_full = True
        else:
            self.position = self.position + 1

    def get_sample(self) -> tuple[list, list, list, list, list] :

        bs = self.position
        state = self.state_buffer[:bs, :]
        action = self.action_buffer[:bs, :]
        reward = self.reward_buffer[:bs]
        next_state = self.next_state_buffer[:bs, :]
        done = self.done_buffer[:bs]
        value = self.value_buffer[:bs]
        logprob = self.logprob_buffer[:bs]
            

        return state, action, reward, next_state, done, value, logprob
    
    @abstractmethod
    def clear(self) -> None:
        pass
    

    def __len__(self) -> int:
        """
        Gets the number of elements in the buffer

        Returns
        -------
        int
            The number of elements currently in the buffer
        """
        if not self.is_full:
            return self.position
        else:
            return self.capacity




class TorchRolloutBuffer(RolloutBuffer):
    def __init__(self, capacity: int, seed: int, state_size: int,
                 action_size: int, device: torch.device) -> None:
        
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

        self.value_buffer = torch.FloatTensor(self.capacity, 1)
        self.value_buffer = self.value_buffer.to(self.device)

        self.logprob_buffer = torch.FloatTensor(self.capacity, 1)
        self.logprob_buffer = self.logprob_buffer.to(self.device)

    def clear(self):
        """free the trajectories collected by last policy"""
        self.state_buffer.zero_()
        self.next_state_buffer.zero_()
        self.action_buffer.zero_()
        self.reward_buffer.zero_()
        self.done_buffer.zero_()
        self.value_buffer.zero_()
        self.logprob_buffer.zero_()
        self.position = 0





