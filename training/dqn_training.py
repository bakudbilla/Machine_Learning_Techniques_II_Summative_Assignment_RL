import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from environment.custom_env import LivestockMonitoringEnv


# Custom callback to track and print average total reward
class RewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if len(self.locals["infos"]) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
        return True

    def _on_training_end(self):
        if self.episode_rewards:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            print(f"\nAverage total reward over {len(self.episode_rewards)} episodes: {avg_reward:.2f}")
        else:
            print("No episodes completed during training.")


def train_dqn(total_timesteps=150_000):
    env = LivestockMonitoringEnv()

    # Creating directories to save files
    save_path = "models/dqn"
    log_path = os.path.join(save_path, "tensorboard_logs")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # setting up TensorBoard logger
    new_logger = configure(log_path, ["stdout", "tensorboard"])

    # Initializing model
    model = DQN(
        "MlpPolicy",
    env,
    learning_rate=5e-5,             
    buffer_size=10_000,             
    batch_size=128,                  
    gamma=0.95,                      
    train_freq=4,
    target_update_interval=300,      
    exploration_fraction=0.15,       
    exploration_final_eps=0.01,      
    policy_kwargs=dict(
        net_arch=[256, 256]          
    ),
    tensorboard_log=log_path
    )

    model.set_logger(new_logger)

    # Evaluation and reward tracking callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    reward_callback = RewardTrackingCallback()

    # Training model
    model.learn(total_timesteps=total_timesteps, 
                callback=[eval_callback, 
                          reward_callback])
    model.save(f"{save_path}/livestock_dqn")

    # Print final average reward 
    if reward_callback.episode_rewards:
        avg_reward = sum(reward_callback.episode_rewards) / len(reward_callback.episode_rewards)
        print(f"Final average reward: {avg_reward:.2f}")
    else:
        print("No episodes completed during training.")

    print("Training complete. Model saved.")


if __name__ == "__main__":
    train_dqn()
