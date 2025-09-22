import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()
        # 4 lanes → North, East, South, West
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.int32)
        # 2 actions → 0 = NS green, 1 = EW green
        self.action_space = spaces.Discrete(2)
        self.state = np.array([10, 10, 10, 10])
        self.step_count = 0

    def reset(self):
        self.state = np.array([10, 10, 10, 10])
        self.step_count = 0
        return self.state

    def step(self, action):
        cars = np.random.randint(0, 5, size=4)
        self.state += cars
        if action == 0:  # NS green
            self.state[0] = max(0, self.state[0] - np.random.randint(5, 15))
            self.state[2] = max(0, self.state[2] - np.random.randint(5, 15))
        else:            # EW green
            self.state[1] = max(0, self.state[1] - np.random.randint(5, 15))
            self.state[3] = max(0, self.state[3] - np.random.randint(5, 15))

        self.step_count += 1
        reward = -np.sum(self.state)  # negative = more cars waiting
        done = self.step_count >= 50
        return self.state, reward, done, {}
