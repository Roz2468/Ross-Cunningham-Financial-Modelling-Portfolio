import yfinance as yf
import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO

# Fetch historical stock data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    stock = stock[['Close']]
    stock.dropna(inplace=True)  # Ensure no NaN values
    return stock

# Custom Portfolio Optimization Environment
class PortfolioEnv(gym.Env):
    def __init__(self, stock_data):
        super(PortfolioEnv, self).__init__()
        self.stock_data = stock_data.pct_change().dropna()
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.positions = 0

        # Action space: Continuous values between -1 (sell all) and 1 (buy all)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = 0
        return np.array([self.stock_data.iloc[self.current_step].values[0]])
    
    def step(self, action):
        if self.current_step >= len(self.stock_data) - 1:
            return np.array([self.stock_data.iloc[self.current_step].values[0]]), 0, True, {}
        
        prev_value = self.balance + (self.positions * self.stock_data.iloc[self.current_step].values[0])
        stock_price = self.stock_data.iloc[self.current_step].values[0]
        if stock_price != 0:
            self.positions += action[0] * (self.balance / stock_price)
            self.balance -= action[0] * self.balance
        
        self.current_step += 1
        stock_price_next = self.stock_data.iloc[self.current_step].values[0]
        new_value = self.balance + (self.positions * stock_price_next)
        reward = new_value - prev_value
        done = self.current_step >= len(self.stock_data) - 1
        
        return np.array([stock_price_next]), reward, done, {}

# Main script
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-01-01'
stock_data = get_stock_data(ticker, start_date, end_date)

# Initialize environment
env = PortfolioEnv(stock_data)

# Train RL agent
# PPO (Proximal Policy Optimization) is chosen for its balance between sample efficiency and stability.
# "MlpPolicy" refers to a multi-layer perceptron (MLP) neural network policy, which learns the optimal trading strategy.
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Evaluate agent
done = False
obs = env.reset()
total_reward = 0
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward

# Total reward represents the cumulative portfolio return over the entire trading period.
print(f"Total Portfolio Return: {total_reward}")
