import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.custom_env import LivestockMonitoringEnv

class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

def train_reinforce(env, total_episodes=5000, gamma=0.99, lr=1e-3):
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNetwork(obs_size, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(total_episodes):
        obs, _ = env.reset()
        rewards = []
        log_probs = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            obs_v = torch.FloatTensor(np.array(obs).reshape(1, -1))
            probs = policy(obs_v)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            obs, reward, terminated, truncated, _ = env.step(action.item())

            rewards.append(reward)
            log_probs.append(log_prob)

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Calculate policy gradient loss
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}, loss: {loss.item():.3f}, total_reward: {sum(rewards):.2f}")

    # Make sure directory exists before saving
    save_dir = "models/reinforce"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(policy.state_dict(), os.path.join(save_dir, "livestock_reinforce.pt"))
    print(f"Training complete and model saved to {os.path.join(save_dir, 'livestock_reinforce.pt')}")

if __name__ == "__main__":
    env = LivestockMonitoringEnv()
    train_reinforce(env)
