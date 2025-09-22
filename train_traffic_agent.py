# train_traffic_agent.py

import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

# ------------------------
# Traffic Environment
# ------------------------
class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()
        self.num_lanes = 4
        
        # Observation: number of vehicles per lane
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.num_lanes,), dtype=np.int32)
        
        # Action: choose lane to give green signal
        self.action_space = spaces.Discrete(self.num_lanes)
        
        self.state = np.zeros(self.num_lanes, dtype=np.int32)
        self.max_vehicles = 100

    def reset(self, seed=None, options=None):
        self.state = np.zeros(self.num_lanes, dtype=np.int32)
        return self.state, {}

    def step(self, action):
        # Vehicles randomly arrive in all lanes
        self.state += np.random.randint(0, 5, size=self.num_lanes)
        
        # Green signal clears some cars in chosen lane
        self.state[action] = max(0, self.state[action] - np.random.randint(5, 15))
        
        # Clip vehicle counts to max limit
        self.state = np.clip(self.state, 0, self.max_vehicles)
        
        # Reward: negative sum of vehicles (agent tries to minimize traffic)
        reward = -np.sum(self.state)
        
        done = False
        truncated = False
        info = {}
        return self.state, reward, done, truncated, info

# ------------------------
# Train RL Agent
# ------------------------
def train_agent():
    env = TrafficEnv()
    
    # Create PPO agent
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the agent
    print("🚦 Training RL agent...")
    model.learn(total_timesteps=50000)
    
    # Save the trained model
    model.save("traffic_rl_agent")
    print("✅ RL agent trained and saved as 'traffic_rl_agent.zip'")

if __name__ == "__main__":
    train_agent()
