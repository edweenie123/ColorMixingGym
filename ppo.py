from env import ColorMixingEnv, Paint
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import os
from stable_baselines3.common.logger import configure
import argparse
import pandas as pd
import numpy as np

# class CustomCallback(BaseCallback):
#     def __init__(self, log_interval=2048, verbose=0):
#         super(CustomCallback, self).__init__(verbose)
#         self.log_interval = log_interval
#         self.metrics = []

#     def _on_step(self) -> bool:
#         # Only log at intervals of `log_interval`
#         if self.num_timesteps % self.log_interval == 0 or self.num_timesteps == 1:
#             logger = self.model.logger
#             # print(type(logger))
#             latest_log = logger.get_log_dict()

#             metrics = {
#                 "total_timesteps": self.num_timesteps,
#                 "ep_rew_mean": latest_log.get("rollout/ep_rew_mean", np.nan)  # Use np.nan for missing values
#             }
#             self.metrics.append(metrics)
#         return True


def train():
    env = ColorMixingEnv(5, noise_level=0.1)

    # Wrap it for vectorized environments
    vec_env = make_vec_env(lambda: env, n_envs=10)

    # Instantiate the agent
    model = PPO("MlpPolicy", vec_env, verbose=1, device='cuda')

    # use this callback during training
    # callback = CustomCallback()
    # Train the agent

    # new_logger = configure('logs', ["stdout", "csv"])
    # model.set_logger(new_logger)
    model.learn(total_timesteps=100000)
    
    # df = pd.DataFrame(callback.metrics)
    # df.to_csv('metrics.csv', index=False)

    # Save the model
    model.save("color_mixing_ppo_agent")


def test():
    env = ColorMixingEnv(5, noise_level=0.1)
    model = PPO.load("color_mixing_ppo_agent.zip")

    num_episodes = 5  # You can adjust this number
    for _ in range(num_episodes):
        obs, _ = env.reset()
        print('initial state', obs)
        env.render()  # Render the environment at each step
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            print(f"Step: {env.step_count}, Action: {action}, Reward: {reward}")
            env.render()  # Render the environment at each step
        
        # Close the rendering window at the end of each episode (if applicable)
        if hasattr(env, 'close'):
            env.close()

    # for entry in os.listdir('tests/'):
    #     # Construct the full path
    #     full_path = os.path.join('tests/', entry)

    #     # Check if it's a file and not a directory
    #     if not os.path.isfile(full_path):
    #         continue

    #     obs = env.load_state_from_file(full_path)
    #     print('initial state', obs)
    #     env.render()  # Render the environment at each step
    #     done = False
    #     while not done:
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, reward, done, truncated, info = env.step(action)

    #         print(f"Step: {env.step_count}, Action: {action}, Reward: {reward}")
    #         env.render()  # Render the environment at each step


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Example script with train and test modes.")

    # Add arguments
    parser.add_argument("--train", action="store_true", help="Run in training mode")
    parser.add_argument("--test", action="store_true", help="Run in testing mode")

    # Parse arguments
    args = parser.parse_args()

    # Execute the appropriate function based on the arguments
    if args.train:
        train()
    elif args.test:
        test()
    else:
        print("No mode selected, add --train or --test")

if __name__ == "__main__":
    main()

