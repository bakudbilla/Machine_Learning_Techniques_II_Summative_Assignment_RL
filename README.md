# Livestock Monitoring with Reinforcement Learning

This project simulates a drone-based livestock monitoring system using Reinforcement Learning (RL). An agent (drone) learns to navigate a 10x10 grid and locate distressed cows efficiently. Multiple RL algorithms (DQN, PPO, A2C, REINFORCE) are trained and evaluated in a custom Gym environment with interactive visualization using Pygame.

<img width="488" height="570" alt="Screenshot 2025-08-01 130551" src="https://github.com/user-attachments/assets/2627f86f-fa6d-42e4-a97b-5a632b92fa79" />


## Project Features

- Custom Environment: Simulates cows with random movement and distress signals.
- Multiple RL Algorithms: Compare DQN, PPO, A2C, and REINFORCE agents.
- Pygame Visualization: Visualize agent movement and animal states.
- Reward System: Encourages quick detection of distressed animals and penalizes unnecessary movement.
- Evaluation & Generalization: Models are tested on unseen initial positions to assess robustness.

## Project Structure
```
├── environment/
│ └── custom_env.py           # Custom Gym environment
├── training/
│ └── train_dqn.py            # DQN training script
│ └── train_ppo.py            # PPO training script
├── rendering/
│ └── render.py               # Grid renderer with legend
├── assets/
│ └── cow.png
│ └── distress_cow.png
│ └── drone.png
├── main.py                   # Run environment and agent
├── test_generalization.py    # Evaluate agent on unseen positions
├── models/
│ └── dqn/                    # Trained models and logs
├── README.md
```

## Algorithms and Architectures

### Deep Q-Network (DQN)

- MLP policy with architecture: [256, 256]
- Experience Replay Buffer: 10,000
- Target network update: every 300 steps
- Exploration: ε-greedy with final ε = 0.01
- Training hyperparameters:
  - learning_rate = 5e-5
  - gamma = 0.95
  - batch_size = 128
  - train_freq = 4

## How to Run

### 1. Clone the repository
```
git clone https://github.com/your-username/livestock-monitoring-rl.git
cd livestock-monitoring-rl
