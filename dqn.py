from env import ColorMixingEnv, Paint
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

# ! DQN DOES NOT WORK WITH multi-descrete action space...

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


# Check if the environment follows the Gym interface
check_env(env)

# Initialize the DQN agent
model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000, learning_rate=1e-3)

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("color_mixing_dqn_model")