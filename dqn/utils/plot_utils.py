"""Plotting utilities for visualizing training progress."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_training_results(
    rewards: List[float],
    eval_rewards: List[float],
    eval_episodes: List[int],
    save_dir: str,
    window_size: int = 100,
    title_prefix: str = "",
):
    """Plot training and evaluation rewards.

    Args:
        rewards: List of episode rewards during training
        eval_rewards: List of evaluation rewards
        eval_episodes: List of episode numbers when evaluations occurred
        save_dir: Directory to save plots
        window_size: Window size for moving average
        title_prefix: Prefix for plot title (e.g., agent type)
    """
    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Calculate moving average of training rewards
    moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot training rewards
    ax1.plot(rewards, alpha=0.3, color="blue", label="Raw Rewards")
    ax1.plot(
        np.arange(window_size - 1, len(rewards)),
        moving_avg,
        color="red",
        label=f"{window_size}-Episode Moving Average",
    )
    ax1.set_ylabel("Training Reward")
    ax1.set_title(f"{title_prefix}Training Rewards")
    ax1.legend()
    ax1.grid(True)

    # Plot evaluation rewards
    ax2.plot(
        eval_episodes,
        eval_rewards,
        color="green",
        marker="o",
        label="Evaluation Rewards",
    )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Evaluation Reward")
    ax2.set_title(f"{title_prefix}Evaluation Rewards")
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_dir / "training_results.png")
    plt.close()


def plot_reward_comparison(rewards_dict: dict, save_dir: str, window_size: int = 100):
    """Plot reward comparison between different agents.

    Args:
        rewards_dict: Dictionary mapping agent names to their reward histories
        save_dir: Directory to save plots
        window_size: Window size for moving average
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for agent_name, rewards in rewards_dict.items():
        # Calculate moving average
        moving_avg = np.convolve(
            rewards, np.ones(window_size) / window_size, mode="valid"
        )
        plt.plot(
            np.arange(window_size - 1, len(rewards)),
            moving_avg,
            label=f"{agent_name} ({window_size}-Episode MA)",
        )

    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Performance Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / "agent_comparison.png")
    plt.close()
