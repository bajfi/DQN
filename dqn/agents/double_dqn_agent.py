from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from ..models.q_network import QNetwork
from .base_agent import BaseAgent


class DoubleDQNAgent(BaseAgent):
    """Double DQN Agent implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"

    def _init_networks(self):
        """Initialize policy and target networks."""
        self.policy_net = QNetwork(
            self.state_size, self.action_size, self.config.get("hidden_size", 128)
        ).to(self.device)

        self.target_net = QNetwork(
            self.state_size, self.action_size, self.config.get("hidden_size", 128)
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def _preprocess_state(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Preprocess state for network input.

        Args:
            state: Raw state from environment

        Returns:
            Preprocessed state tensor
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        return state.to(self.device)

    @torch.no_grad()
    def select_action(self, state: Union[np.ndarray, torch.Tensor]) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        if torch.rand(1) > self.epsilon:
            state = self._preprocess_state(state)
            with autocast(device_type=self.amp_device_type):
                q_values = self.policy_net(state)
            return q_values.argmax().item()
        return torch.randint(self.action_size, (1,)).item()

    def compute_loss(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute the Double DQN loss for a batch of experiences.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)

        Returns:
            Computed loss
        """
        states, actions, rewards, next_states, dones = batch

        with autocast(device_type=self.amp_device_type):
            # Get current Q values
            current_q = self.policy_net(states).gather(1, actions)

            # Compute target Q values using Double DQN
            with torch.no_grad():
                # Select actions using policy network
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                # Evaluate actions using target network
                next_q = self.target_net(next_states).gather(1, next_actions)
                target_q = rewards + self.gamma * next_q * (1 - dones.float())

            # Use Huber loss for better stability
            loss = F.smooth_l1_loss(current_q, target_q)

        return loss
