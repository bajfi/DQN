from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from ..models.q_network import QNetwork
from .base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """Optimized DQN Agent with mixed precision training and gradient clipping."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler()  # For mixed precision training
        self.amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"

    def _init_networks(self):
        """Initialize policy and target networks with optimized settings."""
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
        if state.dim() == 1:
            state = state.unsqueeze(0)
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
            self.policy_net.eval()  # Set to eval mode for inference
            with autocast(device_type=self.amp_device_type):
                q_values = self.policy_net(state)
            self.policy_net.train()  # Set back to train mode
            return q_values.argmax().item()
        return torch.randint(self.action_size, (1,)).item()

    def compute_loss(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute DQN loss with mixed precision and Huber loss for stability.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)

        Returns:
            Computed loss
        """
        states, actions, rewards, next_states, dones = batch

        with autocast(device_type=self.amp_device_type):
            # Get current Q values
            current_q = self.policy_net(states).gather(1, actions)

            # Compute target Q values
            with torch.no_grad():
                # Double Q-learning: use policy net to select actions
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
                target_q = rewards + self.gamma * next_q * (1 - dones.float())

            # Use Huber loss for better stability
            loss = F.smooth_l1_loss(current_q, target_q)

        return loss

    def update(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """Update network parameters with mixed precision training.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)

        Returns:
            Loss value
        """
        self.optimizer.zero_grad()

        # Compute loss with mixed precision
        loss = self.compute_loss(batch)

        # Scale loss and backpropagate
        self.scaler.scale(loss).backward()

        # Clip gradients for stability
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        # Update weights with gradient scaling
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    @torch.no_grad()
    def update_target_network(self):
        """Update target network using Polyak averaging for smoother updates."""
        tau = self.config.get("target_update_tau", 0.005)
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.lerp_(policy_param.data, tau)
