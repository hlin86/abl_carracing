import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import matplotlib.pyplot as plt

# Define the neural network for the policy and value function
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        action_probs = torch.tanh(self.actor(x))  # Assuming continuous action space
        state_value = self.critic(x)
        return action_probs, state_value

# PPO Agent
class PPO:
    def __init__(self, env, gamma=0.99, lam=0.95, epsilon=0.2, lr=3e-4, epochs=10, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.policy = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def compute_gae(self, rewards, values, next_values, masks):
        gae = 0
        returns = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * masks[t] - values[t]
            gae = delta + self.gamma * self.lam * masks[t] * gae
            returns.insert(0, gae + values[t])
        return returns

    def update(self, states, actions, log_probs_old, returns, advantages):
        for _ in range(self.epochs):
            for i in range(0, len(states), self.batch_size):
                batch_states = torch.tensor(states[i:i+self.batch_size], dtype=torch.float32).to(device)
                batch_actions = torch.tensor(actions[i:i+self.batch_size], dtype=torch.float32).to(device)
                batch_log_probs_old = torch.tensor(log_probs_old[i:i+self.batch_size], dtype=torch.float32).to(device)
                batch_returns = torch.tensor(returns[i:i+self.batch_size], dtype=torch.float32).to(device)
                batch_advantages = torch.tensor(advantages[i:i+self.batch_size], dtype=torch.float32).to(device)

                # Forward pass
                action_probs, state_values = self.policy(batch_states)
                dist = Normal(action_probs, torch.ones_like(action_probs) * 0.1)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().mean()

                # Compute ratios
                ratios = torch.exp(log_probs - batch_log_probs_old)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (batch_returns - state_values).pow(2).mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, timesteps):
        state = self.env.reset()
        done = False
        episode_rewards = []
        episode_lengths = []
        states, actions, rewards, log_probs, values, next_values, masks = [], [], [], [], [], [], []

        while timesteps > 0:
            # Collect experience
            for _ in range(2048):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action_probs, value = self.policy(state_tensor)
                dist = Normal(action_probs, torch.ones_like(action_probs) * 0.1)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

                next_state, reward, done, _, _ = self.env.step(action.cpu().numpy())
                mask = 1 - done

                states.append(state)
                actions.append(action.cpu().numpy())
                rewards.append(reward)
                log_probs.append(log_prob.item())
                values.append(value.item())
                next_values.append(self.policy(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device))[1].item())
                masks.append(mask)

                state = next_state
                timesteps -= 1

                if done:
                    episode_rewards.append(sum(rewards))
                    episode_lengths.append(len(rewards))
                    state = self.env.reset()
                    rewards.clear()
                    log_probs.clear()
                    values.clear()
                    next_values.clear()
                    masks.clear()

            # Compute GAE and returns
            returns = self.compute_gae(rewards, values, next_values, masks)
            advantages = np.array(returns) - np.array(values)

            # Update policy
            self.update(states, actions, log_probs, returns, advantages)

            # Clear memory
            states.clear()
            actions.clear()
            rewards.clear()
            log_probs.clear()
            values.clear()
            next_values.clear()
            masks.clear()

        return episode_rewards, episode_lengths

# Initialize environment and agent
env = gym.make('CarRacing-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ppo = PPO(env)

# Train the agent
episode_rewards, episode_lengths = ppo.train(1000000)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title('Episode Rewards Over Time')
plt.subplot(1, 2, 2)
plt.plot(episode_lengths)
plt.title('Episode Lengths Over Time')
plt.show()
