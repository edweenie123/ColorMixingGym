from cm_env import ColorMixingEnv, Paint
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import argparse

def train():
    env = ColorMixingEnv(
        beakers=[
            Paint((255, 0, 0), 100),
            Paint((0, 255, 0), 100),
            Paint((0, 0, 255), 100),
            Paint((0, 0, 0), 0) # empty beaker
        ],
        target_color=(128, 128, 0),  # Olive green
        target_amount=150
    )

    # Wrap it for vectorized environments
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Instantiate the agent
    model = PPO("MlpPolicy", vec_env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=15000)

    # Save the model
    model.save("color_mixing_ppo_agent")

def test():
    env = ColorMixingEnv(4)

    # model = PPO.load("color_mixing_ppo_agent.zip")

    # num_episodes = 5  # You can adjust this number
    # for episode in range(num_episodes):
    #     obs, _ = env.reset()
    #     print('initial state', obs)
    #     env.render()  # Render the environment at each step
    #     done = False
    #     while not done:
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, reward, done, truncated, info = env.step(action)
    #         print(f"Step: {env.step_count}, Action: {action}, Reward: {reward}")
    #         env.render()  # Render the environment at each step
        
    #     # Close the rendering window at the end of each episode (if applicable)
    #     if hasattr(env, 'close'):
    #         env.close()

    # obs = env.load_state_from_file('tests/teal.txt')

    model = PPO.load("color_mixing_ppo_agent.zip")

    for entry in os.listdir('tests/'):
        # Construct the full path
        full_path = os.path.join('tests/', entry)

        # Check if it's a file and not a directory
        if not os.path.isfile(full_path):
            continue

        obs = env.load_state_from_file(full_path)
        print('initial state', obs)
        env.render()  # Render the environment at each step
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            print(f"Step: {env.step_count}, Action: {action}, Reward: {reward}")
            env.render()  # Render the environment at each step



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




# Load the trained model
