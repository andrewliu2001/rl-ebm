"""
Custom OpenAI Gym environment for RL finetuning
"""

import gym
from gym import spaces
import numpy as np

class MoleculeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N)  # Adjust N to your needs
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(HEIGHT, WIDTH, CHANNELS), 
                                            dtype=np.uint8)

    def step(self, action):
        # Implement your step function here
        return observation, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        return observation  # return initial observation

    def render(self, mode='human'):
        # Implement rendering (optional)
        pass

    def close(self):
        # Perform any necessary cleanup (optional)
        pass
