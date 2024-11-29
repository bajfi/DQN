"""Default configuration for DQN agents."""

DEFAULT_CONFIG = {
    # Network parameters
    "hidden_size": 128,
    # Training parameters
    "batch_size": 64,
    "learning_rate": 1e-3,
    "gamma": 0.99,  # Discount factor
    # Memory parameters
    "memory_size": 100_000,
    # Exploration parameters
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    # Target network update frequency
    "target_update": 10,
    "target_update_tau": 0.005,  # For soft updates
    # Training duration
    "num_episodes": 1000,
    # Environment parameters
    "env_name": "CartPole-v1",
    # Checkpoint parameters
    "checkpoint_freq": 100,  # Save model every N episodes
    "checkpoint_dir": "checkpoints",
}
