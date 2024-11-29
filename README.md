# Deep Q-Network (DQN) Implementation

A comprehensive implementation of Deep Q-Network (DQN) and its variants in PyTorch, featuring real-time training visualization and system monitoring.

## Features

- 🧠 Multiple DQN variants:
  - Vanilla DQN
  - Double DQN
  - Dueling DQN
- 📊 Real-time training visualization:
  - Training statistics
  - Progress tracking
  - System resource monitoring (CPU/GPU usage)
- 🔄 Experience replay buffer
- 🎯 Target network for stable training
- 📈 Automatic model checkpointing
- 📉 Training history plotting
- 🎮 Support for various Gymnasium environments

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/DQN.git
cd DQN
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
DQN/
├── dqn/                    # Main package directory
│   ├── agents/            # DQN agent implementations
│   │   ├── base_agent.py
│   │   ├── dqn_agent.py
│   │   ├── double_dqn_agent.py
│   │   └── dueling_dqn_agent.py
│   ├── configs/           # Configuration files
│   │   └── default_config.py
│   ├── models/            # Neural network architectures
│   │   └── q_network.py
│   ├── utils/             # Utility functions
│   │   ├── plot_utils.py
│   │   └── replay_buffer.py
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── requirements.txt       # Project dependencies
├── setup.py              # Package setup file
└── README.md             # This file
```

## Usage

### Training

To train a DQN agent:

```bash
python -m dqn.train --agent [dqn|double_dqn|dueling_dqn] --env [environment_name]
```

Example:

```bash
python -m dqn.train --agent dueling_dqn --env CartPole-v1
```

### Evaluation

To evaluate a trained agent:

```bash
python -m dqn.evaluate [model_path] --env [environment_name] --episodes [num_episodes]
```

Example:

```bash
python -m dqn.evaluate saved_models/best_dueling_dqn_model.pth --env CartPole-v1 --episodes 10
```

## Configuration

You can modify the training parameters in `dqn/configs/default_config.py`. Key parameters include:

- Learning rate
- Batch size
- Memory size
- Discount factor (gamma)
- Epsilon parameters for exploration
- Network architecture
- Training intervals

## Training Visualization

The training script provides real-time visualization with:

- Left panel:
  - Current episode progress
  - Episode rewards
  - Moving average reward
  - Best reward achieved
  - Latest loss value

- Right panel:
  - Training progress bar
  - CPU usage
  - Memory usage
  - GPU usage and memory (if available)

## Results

Models are automatically saved during training:

- `best_{agent_type}_model.pth`: Best performing model
- `latest_{agent_type}_model.pth`: Most recent model
- `final_{agent_type}_model.pth`: Model after training completion

Training plots are saved in the specified output directory, showing:

- Episode rewards
- Moving average rewards
- Evaluation rewards

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [DeepMind's DQN Paper](https://www.nature.com/articles/nature14236)
- [OpenAI's Gymnasium](https://gymnasium.farama.org/)
- [PyTorch](https://pytorch.org/)
- [Rich](https://rich.readthedocs.io/) for terminal visualization
