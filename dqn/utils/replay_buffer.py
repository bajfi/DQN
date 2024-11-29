from typing import Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


class ReplayBuffer:
    """A highly optimized replay buffer using numpy arrays."""

    def __init__(self, capacity: int, state_dim: int, device: str = "cuda"):
        """Initialize the replay buffer with pre-allocated arrays.

        Args:
            capacity: Maximum number of experiences to store
            state_dim: Dimension of the state space
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        # Pre-allocate numpy arrays for better performance
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(
        self,
        state: NDArray[np.float32],
        action: int,
        reward: float,
        next_state: NDArray[np.float32],
        done: bool,
    ) -> None:
        """Add experience to buffer with vectorized operations.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Update experience using numpy's efficient array operations
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[Tensor, ...]:
        """Sample a batch of experiences efficiently.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        # Sample indices using numpy's efficient random sampling
        indices = np.random.choice(self.size, batch_size, replace=False)

        # Convert to tensors efficiently using from_numpy
        states = torch.from_numpy(self.states[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
