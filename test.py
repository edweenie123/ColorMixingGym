from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode='human')
# model = PPO(policy = "MlpPolicy",env =  env, verbose=1)
# model.learn(total_timesteps=25000)

# model.save("ppo_cartpole")  # saving the model to ppo_cartpole.zip
model = PPO.load("ppo_cartpole")  # loading the model from ppo_cartpole.zip

obs, _ = env.reset()
# print(obs)
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()