from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.custom_env import LivestockMonitoringEnv

def train_ppo(total_timesteps=100_000):
    env = LivestockMonitoringEnv()

    # Paths for saving model and logs
    save_path = "models/ppo"
    log_path = os.path.join(save_path, "tensorboard_logs")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Configure logger
    custom_logger = configure(log_path, ["stdout", "tensorboard"])

    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        gamma=0.99,
        tensorboard_log=log_path  # Needed for naming logs
    )
    model.set_logger(custom_logger)

    # Evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=10_000,
        deterministic=True,
        render=False
    )

    # Training
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"{save_path}/livestock_ppo")

    print("PPO training complete.")

if __name__ == "__main__":
    train_ppo()
