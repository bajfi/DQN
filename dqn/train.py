#!/usr/bin/env python3
"""Training script for DQN agents."""

from enum import Enum
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import psutil
import torch
import typer
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from dqn.agents.double_dqn_agent import DoubleDQNAgent
from dqn.agents.dqn_agent import DQNAgent
from dqn.agents.dueling_dqn_agent import DuelingDQNAgent
from dqn.configs.default_config import DEFAULT_CONFIG
from dqn.utils.plot_utils import plot_training_results

# Initialize Typer app and Rich console
app = typer.Typer(
    help="Train DQN agents",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def get_system_usage():
    """Get current CPU and GPU usage.

    Returns:
        dict: System usage statistics
    """
    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    memory_percent = memory.percent

    # Get GPU usage if available
    gpu_stats = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                gpu_stats[i] = {
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated": torch.cuda.memory_allocated(i)
                    / (1024**3),  # GB
                    "memory_reserved": torch.cuda.memory_reserved(i) / (1024**3),  # GB
                    "max_memory": torch.cuda.get_device_properties(i).total_memory
                    / (1024**3),  # GB
                }
            except Exception:
                continue

    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "gpu_stats": gpu_stats,
    }


def create_system_usage_table(usage_stats: dict) -> Table:
    """Create a table showing system resource usage.

    Args:
        usage_stats: Dictionary containing system usage statistics

    Returns:
        Table: Rich table containing system usage information
    """
    table = Table(title="System Usage", show_header=True, header_style="bold magenta")
    table.add_column("Resource", style="cyan")
    table.add_column("Usage", justify="right", style="green")

    # Add CPU and Memory usage
    table.add_row("CPU Usage", f"{usage_stats['cpu_percent']}%")
    table.add_row("Memory Usage", f"{usage_stats['memory_percent']}%")

    # Add GPU usage if available
    if usage_stats["gpu_stats"]:
        for gpu_id, gpu_info in usage_stats["gpu_stats"].items():
            name = gpu_info["name"]
            allocated = gpu_info["memory_allocated"]
            reserved = gpu_info["memory_reserved"]
            max_mem = gpu_info["max_memory"]

            table.add_row(
                f"GPU {gpu_id} ({name})",
                f"Used: {allocated:.1f}GB | "
                f"Reserved: {reserved:.1f}GB | "
                f"Total: {max_mem:.1f}GB",
            )

    return table


def create_layout() -> Layout:
    """Create the layout for the live display."""
    layout = Layout()

    # Split into header and content
    layout.split(
        Layout(name="header", size=3),
        Layout(name="content"),
    )

    # Split content into left and right sections with 1:2 ratio
    layout["content"].split_row(
        Layout(name="left_section", ratio=1),  # Statistics panel (1 part)
        Layout(
            name="right_section", ratio=2
        ),  # Progress and system usage panel (2 parts)
    )

    return layout


class AgentType(str, Enum):
    """Available agent types."""

    DQN = "dqn"
    DOUBLE_DQN = "double_dqn"
    DUELING_DQN = "dueling_dqn"


def get_agent(agent_type: AgentType, env, config: dict):
    """Get the appropriate agent based on type."""
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agents = {
        AgentType.DQN: DQNAgent,
        AgentType.DOUBLE_DQN: DoubleDQNAgent,
        AgentType.DUELING_DQN: DuelingDQNAgent,
    }

    return agents[agent_type](state_dim, n_actions, config)


def create_stats_table(
    episode: int,
    total_episodes: int,
    episode_reward: float,
    avg_reward: float,
    best_reward: float,
    eval_reward: Optional[float] = None,
    loss: Optional[float] = None,
) -> Table:
    """Create a table showing training statistics and loss.

    Args:
        episode: Current episode number
        total_episodes: Total number of episodes
        episode_reward: Current episode reward
        avg_reward: Moving average reward
        best_reward: Best reward so far
        eval_reward: Latest evaluation reward
        loss: Latest training loss

    Returns:
        Table containing training statistics
    """
    table = Table(show_header=False, pad_edge=False, box=None)

    # Add episode progress
    table.add_row(
        Text("Episode", style="cyan"),
        Text(f"{episode}/{total_episodes}", style="green"),
        Text(f"({episode/total_episodes*100:.1f}%)", style="blue"),
    )

    # Add rewards
    table.add_row(
        Text("Current Reward", style="cyan"),
        Text(f"{episode_reward:.2f}", style="green"),
    )
    table.add_row(
        Text("Moving Average", style="cyan"), Text(f"{avg_reward:.2f}", style="green")
    )
    table.add_row(
        Text("Best Reward", style="cyan"), Text(f"{best_reward:.2f}", style="green")
    )

    # Add evaluation reward if available
    if eval_reward is not None:
        table.add_row(
            Text("Eval Reward", style="cyan"), Text(f"{eval_reward:.2f}", style="green")
        )

    # Add loss if available
    if loss is not None:
        table.add_row(
            Text("Latest Loss", style="cyan"), Text(f"{loss:.4f}", style="yellow")
        )

    return table


def create_progress_section(progress: Progress, usage_stats: dict) -> Panel:
    """Create a section combining progress bar and system usage.

    Args:
        progress: Rich progress bar instance
        usage_stats: System usage statistics

    Returns:
        Panel containing progress and system usage
    """
    # Create system usage table
    system_table = Table(show_header=True, header_style="bold magenta", padding=(0, 2))
    system_table.add_column("Resource", style="cyan")
    system_table.add_column("Usage", justify="right", style="green")

    # Add CPU and Memory usage
    system_table.add_row("CPU Usage", f"{usage_stats['cpu_percent']}%")
    system_table.add_row("Memory Usage", f"{usage_stats['memory_percent']}%")

    # Add GPU usage if available
    if usage_stats["gpu_stats"]:
        for gpu_id, gpu_info in usage_stats["gpu_stats"].items():
            name = gpu_info["name"]
            allocated = gpu_info["memory_allocated"]
            reserved = gpu_info["memory_reserved"]
            max_mem = gpu_info["max_memory"]

            system_table.add_row(
                f"GPU {gpu_id} ({name})",
                f"Used: {allocated:.1f}GB / Reserved: {reserved:.1f}GB / Total: {max_mem:.1f}GB",
            )

    # Create a Group to combine progress and system usage
    group = Group(
        Panel(progress, title="Training Progress", border_style="blue"),
        "",  # Empty line for spacing
        Panel(system_table, title="System Resources", border_style="blue"),
    )

    return Panel(group, title="Training Status", border_style="blue")


def train(
    env,
    eval_env,
    agent,
    num_episodes: int,
    eval_interval: int,
    eval_episodes: int,
    save_dir: Path,
    window_size: int = 100,
):
    """Train the agent."""
    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    best_reward = float("-inf")
    episode_rewards = []
    eval_rewards = []
    eval_episodes_list = []

    # Create progress bars
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,  # Make sure progress bar expands to fill width
    )

    episode_progress = progress.add_task("[cyan]Training Progress", total=num_episodes)

    # Create layout
    layout = create_layout()
    layout["header"].update(
        Panel(
            f"Training {agent.__class__.__name__} on {env.unwrapped.spec.id}",
            style="bold blue",
        )
    )

    with Live(layout, refresh_per_second=4, console=console):
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False

            while not (done or truncated):
                # Select and perform action
                action = agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)

                # Store transition and train
                agent.store_experience(state, action, reward, next_state, done)
                loss = agent.train_step()

                state = next_state
                episode_reward += reward

            episode_rewards.append(episode_reward)
            progress.update(episode_progress, advance=1)

            # Calculate moving average
            recent_rewards = (
                episode_rewards[-window_size:]
                if len(episode_rewards) >= window_size
                else episode_rewards
            )
            avg_reward = np.sum(recent_rewards) / len(recent_rewards)

            # Evaluate if needed
            eval_reward = None
            if episode % eval_interval == 0:
                eval_reward = evaluate(agent, eval_env, eval_episodes)
                eval_rewards.append(eval_reward)
                eval_episodes_list.append(episode)

                # Save best model
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    model_name = f"best_{agent.__class__.__name__.lower()}_model.pth"
                    save_path = save_dir / model_name
                    agent.save(save_path)
                    console.print(
                        f"[green]New best model saved to {save_path} with reward: {best_reward:.2f}"
                    )

                # Save latest model
                latest_model_name = (
                    f"latest_{agent.__class__.__name__.lower()}_model.pth"
                )
                latest_save_path = save_dir / latest_model_name
                agent.save(latest_save_path)

            # Create and update statistics table
            stats_table = create_stats_table(
                episode + 1,
                num_episodes,
                episode_reward,
                avg_reward,
                best_reward,
                eval_reward,
                loss,
            )

            # Get system usage and create progress section
            usage_stats = get_system_usage()
            progress_section = create_progress_section(progress, usage_stats)

            # Update layout with side-by-side panels
            layout["left_section"].update(
                Panel(
                    stats_table, title="Training Statistics & Loss", border_style="blue"
                )
            )
            layout["right_section"].update(progress_section)

    # Save final model
    final_model_name = f"final_{agent.__class__.__name__.lower()}_model.pth"
    final_save_path = save_dir / final_model_name
    agent.save(final_save_path)
    console.print(f"[green]Final model saved to {final_save_path}")

    # Plot results
    plot_training_results(
        rewards=episode_rewards,
        eval_rewards=eval_rewards,
        eval_episodes=eval_episodes_list,
        save_dir=save_dir,
        title_prefix=f"{agent.__class__.__name__} - ",
        window_size=window_size,
    )

    return episode_rewards, eval_rewards


def evaluate(agent, env, num_episodes: int = 10) -> float:
    """Evaluate the agent."""
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

    return np.sum(rewards) / len(rewards)


@app.command()
def main(
    agent_type: AgentType = typer.Option(
        AgentType.DQN,
        "--agent",
        "-a",
        help="Type of DQN agent to use",
        case_sensitive=False,
    ),
    env_name: str = typer.Option(
        DEFAULT_CONFIG["env_name"],
        "--env",
        "-e",
        help="Gymnasium environment to use",
    ),
    episodes: int = typer.Option(
        1000,
        "--episodes",
        "-n",
        help="Number of episodes to train",
        min=1,
    ),
    eval_interval: int = typer.Option(
        100,
        "--eval-interval",
        help="Episodes between evaluations",
        min=1,
    ),
    eval_episodes: int = typer.Option(
        10,
        "--eval-episodes",
        help="Number of episodes for evaluation",
        min=1,
    ),
    save_dir: Path = typer.Option(
        "results",
        "--save-dir",
        "-s",
        help="Directory to save models and plots",
        dir_okay=True,
        file_okay=False,
    ),
    window_size: int = typer.Option(
        100,
        "--window-size",
        "-w",
        help="Window size for moving average in plots",
        min=1,
    ),
):
    """Train a DQN agent in a Gymnasium environment."""

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create environments
    try:
        env = gym.make(env_name)
        eval_env = gym.make(env_name)
    except Exception as e:
        console.print(f"[red]Error creating environment '{env_name}': {str(e)}")
        raise typer.Exit(1)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[yellow]Using device: {device}")

    try:
        # Create agent
        agent = get_agent(agent_type, env, DEFAULT_CONFIG)
        console.print(f"[green]Created {agent.__class__.__name__}")

        # Train agent
        train(
            env=env,
            eval_env=eval_env,
            agent=agent,
            num_episodes=episodes,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            save_dir=save_dir,
            window_size=window_size,
        )

    except Exception as e:
        console.print(f"[red]Error during training: {str(e)}")
        raise typer.Exit(1)

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    app()
