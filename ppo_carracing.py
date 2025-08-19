import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt

# Wrap environment to log per-episode data
env = Monitor(gym.make("CarRacing-v3", continuous=False), filename="monitor_logs.csv")
vec_env = DummyVecEnv([lambda: env])

# Initialize model with TensorBoard logging
model = PPO("CnnPolicy", vec_env, verbose=1, tensorboard_log="./logs/", stats_window_size=20)

model.learn(total_timesteps=100_000)

# -- Plotting manually after training --
results = load_results("monitor_logs.csv")
timesteps, rewards = ts2xy(results, "timesteps")
plt.plot(timesteps, rewards)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Training Reward Over Time")
plt.show()
