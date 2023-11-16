import gym
from stable_baselines3 import PPO

from env import MoleculeEnv

env = MoleculeEnv()

# Initialize PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Test the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
      obs = env.reset()