# import time
# import gymnasium as gym
# import pygame
# from environment.custom_env import LivestockMonitoringEnv
# from environment.rendering import Renderer
# import random

# def main():
#     env = LivestockMonitoringEnv()
#     renderer = Renderer(env)
#     obs = env.reset()

#     done = False
#     while not done:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 renderer.close()

#         action = env.action_space.sample()  # Random action
#         obs, reward, done, info = env.step(action)
#         renderer.render()
#         time.sleep(0.2)  # Slow down to see moves

#     renderer.close()

# if __name__ == "__main__":
#     main()
# import time
# import pygame
# import argparse
# from environment.custom_env import LivestockMonitoringEnv
# from environment.rendering import Renderer
# from stable_baselines3 import DQN, PPO, A2C

# # Mapping model names to their SB3 classes and paths
# MODEL_MAP = {
#     "dqn": {"cls": DQN, "path": "models/dqn/livestock_dqn"},
#     "ppo": {"cls": PPO, "path": "models/ppo/livestock_ppo"},
#     "a2c": {"cls": A2C, "path": "models/a2c/livestock_a2c"},
# }

# def main(model_name="dqn", num_episodes=5):
#     if model_name.lower() not in MODEL_MAP:
#         print(f"Unsupported model '{model_name}'. Choose from: {list(MODEL_MAP.keys())}")
#         return

#     env = LivestockMonitoringEnv()
#     renderer = Renderer(env)

#     # Load model class and checkpoint
#     model_cls = MODEL_MAP[model_name]["cls"]
#     model_path = MODEL_MAP[model_name]["path"]
#     model = model_cls.load(model_path, env=env)

#     print(f"Playing using model: {model_name.upper()}")

#     for episode in range(1, num_episodes + 1):
#         obs, info = env.reset()
#         terminated = False
#         truncated = False
#         total_reward = 0
#         step_count = 0

#         while not (terminated or truncated):
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     renderer.close()
#                     return

#             action, _states = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = env.step(action)
#             total_reward += reward
#             step_count += 1

#             renderer.render()
#             time.sleep(0.2)

#         print(f"Episode {episode} finished with total reward: {total_reward:.2f}, steps: {step_count}")

#     renderer.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, default="dqn", help="Model to play: dqn, ppo, or a2c")
#     parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
#     args = parser.parse_args()

#     main(model_name=args.model, num_episodes=args.episodes)
import time
import pygame
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import LivestockMonitoringEnv
from environment.rendering import Renderer

# Reinforce Policy class inside main.py
class ReinforcePolicy(nn.Module):
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

    def predict(self, obs):
        obs = np.array(obs).reshape(1, -1)
        obs_tensor = torch.FloatTensor(obs)
        with torch.no_grad():
            probs = self.forward(obs_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

MODEL_MAP = {
    "dqn": {"cls": DQN, "path": "models/dqn/livestock_dqn"},
    "ppo": {"cls": PPO, "path": "models/ppo/livestock_ppo"},
    "a2c": {"cls": A2C, "path": "models/a2c/livestock_a2c"},
    "reinforce": {"cls": ReinforcePolicy, "path": "models/reinforce/livestock_reinforce.pt"},
}

def main(model_name="dqn", num_episodes=5):
    env = LivestockMonitoringEnv()
    renderer = Renderer(env)

    if model_name == "reinforce":
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n
        model = ReinforcePolicy(obs_size, n_actions)
        model.load_state_dict(torch.load(MODEL_MAP[model_name]["path"]))
        model.eval()
    else:
        model_cls = MODEL_MAP[model_name]["cls"]
        model = model_cls.load(MODEL_MAP[model_name]["path"], env=env)

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        while not (terminated or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    renderer.close()
                    return

            if model_name == "reinforce":
                action = model.predict(obs)
            else:
                action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            renderer.render()
            time.sleep(0.2)

        print(f"Episode {episode} finished with total reward: {total_reward:.2f}, steps: {step_count}")

    renderer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dqn", choices=MODEL_MAP.keys(), help="Model to play with")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    args = parser.parse_args()

    main(model_name=args.model, num_episodes=args.episodes)
