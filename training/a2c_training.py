from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.custom_env import LivestockMonitoringEnv

def train_a2c(total_timesteps=200000):
    env = LivestockMonitoringEnv()

    # Paths for saving model and logs
    save_path = "models/a2c"
    log_path = os.path.join(save_path, "tensorboard_logs")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # setting up logger
    custom_logger = configure(log_path, ["stdout", "tensorboard"])

    # Initializing model
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=7e-4,
        gamma=0.99,
        tensorboard_log=log_path  
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

    # Training  the model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"{save_path}/livestock_a2c")

    print("A2C training complete.")

if __name__ == "__main__":
    train_a2c()
