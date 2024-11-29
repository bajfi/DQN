#!/usr/bin/env python3
"""Evaluation script for trained DQN agents."""

import time
from enum import Enum
from pathlib import Path

import gymnasium as gym
import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from dqn.agents.double_dqn_agent import DoubleDQNAgent
from dqn.agents.dqn_agent import DQNAgent
from dqn.agents.dueling_dqn_agent import DuelingDQNAgent
from dqn.configs.default_config import DEFAULT_CONFIG

# Initialize Typer app and Rich console
app = typer.Typer(
    help="Evaluate trained DQN agents",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


class AgentType(str, Enum):
    """Available agent types."""

    DQN = "dqn"
    DOUBLE_DQN = "double_dqn"
    DUELING_DQN = "dueling_dqn"


def get_agent_class(agent_type: str):
    """Get the agent class based on the agent type string.

    Args:
        agent_type: String identifier of the agent type

    Returns:
        Agent class
    """
    agent_classes = {
        "DQNAgent": DQNAgent,
        "DoubleDQNAgent": DoubleDQNAgent,
        "DuelingDQNAgent": DuelingDQNAgent,
    }
    return agent_classes.get(agent_type)


def get_agent(agent_type: AgentType, env, config: dict):
    """Get the appropriate agent based on type.

    Args:
        agent_type: Type of agent to create
        env: Gymnasium environment
        config: Agent configuration

    Returns:
        Agent instance
    """
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agents = {
        AgentType.DQN: DQNAgent,
        AgentType.DOUBLE_DQN: DoubleDQNAgent,
        AgentType.DUELING_DQN: DuelingDQNAgent,
    }

    return agents[agent_type](state_dim, n_actions, config)


def evaluate_episode(env, agent, render: bool = True, delay: float = 0.01) -> float:
    """Run a single evaluation episode.

    Args:
        env: Gymnasium environment
        agent: Trained agent
        render: Whether to render the environment
        delay: Delay between steps when rendering (seconds)

    Returns:
        Episode reward
    """
    state, _ = env.reset()
    episode_reward = 0
    done = False
    truncated = False

    while not (done or truncated):
        if render:
            env.render()
            time.sleep(delay)

        # Select action
        with torch.no_grad():
            action = agent.select_action(state)

        # Take action
        state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward

    return episode_reward


def display_results(rewards: list[float]) -> None:
    """Display evaluation results in a formatted table.

    Args:
        rewards: List of episode rewards
    """
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Number of Episodes", str(len(rewards)))
    table.add_row("Average Reward", f"{sum(rewards) / len(rewards):.2f}")
    table.add_row("Max Reward", f"{max(rewards):.2f}")
    table.add_row("Min Reward", f"{min(rewards):.2f}")
    table.add_row("Standard Deviation", f"{torch.tensor(rewards).std().item():.2f}")

    console.print(table)


def evaluate(
    env, agent, num_episodes: int = 10, render: bool = True, delay: float = 0.01
) -> list[float]:
    """Evaluate a trained agent.

    Args:
        env: Gymnasium environment
        agent: Trained agent
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        delay: Delay between steps when rendering (seconds)

    Returns:
        List of episode rewards
    """
    # Set agent to evaluation mode
    agent.policy_net.eval()
    rewards = []

    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating agent...", total=num_episodes)

        for episode in range(num_episodes):
            reward = evaluate_episode(env, agent, render, delay)
            rewards.append(reward)
            progress.update(
                task, advance=1, description=f"Episode {episode + 1}/{num_episodes}"
            )

    return rewards


@app.command()
def main(
    model_path: Path = typer.Argument(
        ...,
        help="Path to the trained model file",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    env_name: str = typer.Option(
        DEFAULT_CONFIG["env_name"],
        "--env",
        "-e",
        help="Gymnasium environment to use",
    ),
    episodes: int = typer.Option(
        1,
        "--episodes",
        "-n",
        help="Number of episodes to evaluate",
        min=1,
    ),
    no_render: bool = typer.Option(
        False,
        "--no-render",
        help="Disable environment rendering",
    ),
    delay: float = typer.Option(
        0.02,
        "--delay",
        "-d",
        help="Delay between steps when rendering (seconds)",
        min=0.0,
    ),
):
    """Evaluate a trained DQN agent in a Gymnasium environment."""

    # Create environment
    try:
        env = gym.make(env_name, render_mode="human" if not no_render else None)
    except Exception as e:
        console.print(f"[red]Error creating environment '{env_name}': {str(e)}")
        raise typer.Exit(1)

    try:
        # First load the checkpoint to get the agent type
        console.print(f"[yellow]Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location="cpu")
        agent_type = checkpoint.get("agent_type")

        if not agent_type:
            console.print("[red]Error: Could not determine agent type from saved model")
            raise typer.Exit(1)

        # Get the correct agent class
        AgentClass = get_agent_class(agent_type)
        if not AgentClass:
            console.print(f"[red]Error: Unknown agent type '{agent_type}'")
            raise typer.Exit(1)

        console.print(f"[green]Detected agent type: {agent_type}")

        # Create agent with the correct type
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        agent = AgentClass(state_dim, n_actions, DEFAULT_CONFIG)

        # Load the model state
        try:
            agent.load(model_path)
            console.print(f"[green]Successfully loaded model from {model_path}")
        except Exception as e:
            console.print(f"[red]Error loading model state: {str(e)}")
            raise typer.Exit(1)

        # Evaluate agent
        console.print("\n[cyan]Starting evaluation...[/cyan]")
        rewards = evaluate(
            env=env,
            agent=agent,
            num_episodes=episodes,
            render=not no_render,
            delay=delay,
        )

        # Display results
        console.print("\n[cyan]Evaluation Results:[/cyan]")
        display_results(rewards)

    except Exception as e:
        console.print(f"[red]Error during evaluation: {str(e)}")
        raise typer.Exit(1)

    finally:
        env.close()


if __name__ == "__main__":
    app()
