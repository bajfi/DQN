import torch
import torch.nn as nn
from torch import Tensor


class QNetwork(nn.Module):
    """Q-Network with layer normalization for faster training."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """Initialize the Q-Network with optimized architecture.

        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            hidden_size: Number of neurons in hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            x: Input state tensor

        Returns:
            Q-values for each action
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension

        return self.net(x)


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network with advanced architecture."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """Initialize the Dueling Q-Network.

        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            hidden_size: Number of neurons in hidden layers
        """
        super().__init__()

        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Value stream
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Advantage stream
        self.advantage_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass combining value and advantage streams.

        Args:
            x: Input state tensor

        Returns:
            Q-values for each action
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension

        features = self.feature_net(x)
        value = self.value_net(features)
        advantage = self.advantage_net(features)

        # Combine streams using the dueling formula
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
