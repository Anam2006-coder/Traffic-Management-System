# import gymnasium as gym
# import numpy as np
# from gymnasium import spaces

# class TrafficEnv(gym.Env):
#     def __init__(self):
#         super(TrafficEnv, self).__init__()
#         # 4 lanes → North, East, South, West
#         self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.int32)
#         # 2 actions → 0 = NS green, 1 = EW green
#         self.action_space = spaces.Discrete(2)
#         self.state = np.array([10, 10, 10, 10])
#         self.step_count = 0

#     def reset(self):
#         self.state = np.array([10, 10, 10, 10])
#         self.step_count = 0
#         return self.state

#     def step(self, action):
#         cars = np.random.randint(0, 5, size=4)
#         self.state += cars
#         if action == 0:  # NS green
#             self.state[0] = max(0, self.state[0] - np.random.randint(5, 15))
#             self.state[2] = max(0, self.state[2] - np.random.randint(5, 15))
#         else:            # EW green
#             self.state[1] = max(0, self.state[1] - np.random.randint(5, 15))
#             self.state[3] = max(0, self.state[3] - np.random.randint(5, 15))

#         self.step_count += 1
#         reward = -np.sum(self.state)  # negative = more cars waiting
#         done = self.step_count >= 50
#         return self.state, reward, done, {}

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrafficEnv(gym.Env):
    """
    Custom Traffic Environment for RL:
    - 4 lanes, each lane has a vehicle count (0-100)
    - Action: choose one lane to give green signal
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(TrafficEnv, self).__init__()

        # Observation space: vehicle count per lane
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.int32)
        
        # Action space: choose 1 lane to turn green
        self.action_space = spaces.Discrete(4)

        # Initial state
        self.state = np.zeros(4, dtype=np.int32)
        self.step_count = 0
        self.max_steps = 50  # episode length

    def reset(self, *, seed=None, options=None):
        """
        Reset environment to initial state
        """
        super().reset(seed=seed)
        self.state = np.zeros(4, dtype=np.int32)
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        """
        Take an action (which lane to give green signal)
        """
        # Random vehicles arriving in each lane
        self.state = self.state + np.random.randint(0, 5, size=(4,))

        # Green signal reduces vehicles in selected lane
        self.state[action] = max(0, self.state[action] - np.random.randint(5, 15))

        # Reward: lower total vehicles = better
        reward = -np.sum(self.state)

        # Episode termination
        self.step_count += 1
        done = self.step_count >= self.max_steps
        truncated = done

        info = {}

        return self.state, reward, done, truncated, info

    def render(self):
        """
        Optional: print current lane states
        """
        print(f"Step: {self.step_count}, Vehicle counts: {self.state}")

    def close(self):
        pass
