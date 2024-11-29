from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.optim as optim
from torch import Tensor

from ..utils.replay_buffer import ReplayBuffer


class BaseAgent(ABC):
    """Abstract base class for DQN-based agents."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the base agent.

        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            config: Configuration dictionary
            device: Device to run the agent on
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = device

        # Initialize networks
        self.policy_net = None
        self.target_net = None
        self._init_networks()

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.get("learning_rate", 1e-3)
        )

        # Initialize replay buffer
        self.memory = ReplayBuffer(
            capacity=config.get("memory_size", 10000),
            state_dim=state_size,
            device=device,
        )

        # Training parameters
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.target_update_freq = config.get("target_update", 10)
        self.update_count = 0

    @abstractmethod
    def _init_networks(self) -> None:
        """Initialize policy and target networks."""
        pass

    @abstractmethod
    def compute_loss(self, batch: tuple) -> Tensor:
        """Compute the loss for a batch of experiences.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)

        Returns:
            Computed loss
        """
        pass

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.memory.push(state, action, reward, next_state, done)

    def update(self, batch: tuple) -> float:
        """Update the agent's networks.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)

        Returns:
            Loss value
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_step(self) -> Optional[float]:
        """Perform a single training step.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample experiences
        experiences = self.memory.sample(self.batch_size)

        # Update networks
        loss = self.update(experiences)

        # Update target network if needed
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss

    def update_target_network(self) -> None:
        """Update target network parameters."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        if np.random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
        return np.random.randint(self.action_size)

    def save(self, path: Union[str, Path]) -> None:
        """Save the agent's state.

        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "device": self.device,
            "agent_type": self.__class__.__name__,
        }

        torch.save(checkpoint, path)

    def load(self, path: Union[str, Path]) -> None:
        """Load the agent's state.

        Args:
            path: Path to load the model from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, weights_only=False)

        # Verify agent type
        if checkpoint["agent_type"] != self.__class__.__name__:
            raise ValueError(
                f"Model was trained with {checkpoint['agent_type']}, "
                f"but trying to load into {self.__class__.__name__}"
            )

        # Verify state and action sizes
        if (
            checkpoint["state_size"] != self.state_size
            or checkpoint["action_size"] != self.action_size
        ):
            raise ValueError(
                f"Model dimensions (s:{checkpoint['state_size']}, a:{checkpoint['action_size']}) "
                f"don't match environment (s:{self.state_size}, a:{self.action_size})"
            )

        # Load network states
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load other parameters
        self.config.update(checkpoint["config"])
        self.epsilon = checkpoint["epsilon"]
        self.update_count = checkpoint["update_count"]

        # Set networks to appropriate mode
        self.policy_net.train()
        self.target_net.eval()
