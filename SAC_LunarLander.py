import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:{device}")

# ======================
# 1. Actor & Critic 网络
# ======================

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim, beta):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim, max_action, beta):
        super(ActorNetwork, self).__init__()
        self.max_action = max_action
        self.tiny_positive = 1e-6

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.mu = nn.Linear(fc2_dim, action_dim)
        self.log_std = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        sigma = log_std.exp()

        return mu, sigma
    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)

        if reparameterize:
            raw_action = dist.rsample()
        else:
            raw_action = dist.sample()

        tanh_action = torch.tanh(raw_action)
        scaled_action = tanh_action * self.max_action

        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1 - tanh_action.pow(2) + self.tiny_positive)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return scaled_action, log_prob


# ======================
# 2. Replay Buffer
# ======================
class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.next_state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.done_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.buffer_index = 0

    def add_buffer(self, state, action, reward, next_state, done):
        index = self.buffer_index % self.buffer_size
        self.state_buffer[index] = state
        self.next_state_buffer[index] = next_state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.done_buffer[index] = float(done)
        self.buffer_index += 1

    def sample_buffer(self, batch_size):
        current_buffer_size = min(self.buffer_index, self.buffer_size)
        batch = np.random.choice(current_buffer_size, batch_size, replace=False)
        batch_state = self.state_buffer[batch]
        batch_action = self.action_buffer[batch]
        batch_reward = self.reward_buffer[batch]
        batch_next_state = self.next_state_buffer[batch]
        batch_done = self.done_buffer[batch]
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done


class SACAgent:
    def __init__(self, state_dim, action_dim, buffer_size,
                 alpha, critic_lr, actor_lr, gamma, tau,
                 layer1_dim, layer2_dim, batch_size, max_action):
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size

        # 自动调节 alpha
        self.target_entropy = -action_dim  # SAC 论文推荐
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=actor_lr)

        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

        self.critic_1 = CriticNetwork(state_dim, action_dim, layer1_dim, layer2_dim, critic_lr).to(device)
        self.critic_2 = CriticNetwork(state_dim, action_dim, layer1_dim, layer2_dim, critic_lr).to(device)
        self.target_critic_1 = CriticNetwork(state_dim, action_dim, layer1_dim, layer2_dim, critic_lr).to(device)
        self.target_critic_2 = CriticNetwork(state_dim, action_dim, layer1_dim, layer2_dim, critic_lr).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor = ActorNetwork(state_dim, action_dim, layer1_dim, layer2_dim, max_action, actor_lr).to(device)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action, _ = self.actor.sample_normal(state, reparameterize=False)
        action = action.cpu().detach().numpy()[0]
        return action
    
    def add_buffer(self, state, action, reward, next_state, done):
        self.buffer.add_buffer(state, action, reward, next_state, done)

    # 添加软更新方法
    def update_network_parameters(self):
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self):
        if self.buffer.buffer_index < self.batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample_buffer(self.batch_size)

        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = torch.tensor(action, dtype=torch.float32, device=device)
        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        done = torch.tensor(done, dtype=torch.float32, device=device)

        # 1. 计算 target Q
        with torch.no_grad():
            next_action, next_logp = self.actor.sample_normal(next_state, reparameterize=False)
            q1_target = self.target_critic_1(next_state, next_action)
            q2_target = self.target_critic_2(next_state, next_action)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_logp
            q_target = reward + self.gamma * (1 - done) * q_target

        # 2. 更新 Critic
        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # 3. 更新 Actor
        new_action, logp = self.actor.sample_normal(state, reparameterize=True)
        q1_pi = self.critic_1(state, new_action)
        q2_pi = self.critic_2(state, new_action)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * logp - q_pi).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        self.update_network_parameters()

        # 4. 更新 alpha（自动调节）
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()


# ======================
# 3. 训练主循环
# ======================
env_name = "LunarLanderContinuous-v3"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

REPLAYBUFFER_SIZE = 1000000
max_episodes = 600
max_steps = 300
start_steps = 10000 # 添加随机动作预热步数

agent = SACAgent(state_dim, action_dim, REPLAYBUFFER_SIZE,
                 alpha=0.2, critic_lr=3e-4, actor_lr=3e-4, gamma=0.99, tau=0.005,
                 layer1_dim=256, layer2_dim=256, batch_size=256,
                 max_action=max_action)

REWARD_BUFFER = []
best_reward = -np.inf
total_steps = 0

for episode in range(max_episodes):
    state, _ = env.reset()
    episode_reward = 0.0

    for step in range(max_steps):
        total_steps += 1
        if total_steps < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        scaled_reward = reward * 0.1 # Reward Scaling

        agent.add_buffer(state, action, scaled_reward, next_state, done)
        episode_reward += reward
        state = next_state

        if total_steps >= start_steps:
            agent.update()

        if done:
            break

    REWARD_BUFFER.append(episode_reward)
    avg_reward = np.mean(REWARD_BUFFER[-10:])  # 最近 10 回合平均

    if episode >= 50 and avg_reward > best_reward:
        best_reward = avg_reward
        current_path = os.path.dirname(os.path.realpath(__file__))
        save_dir = os.path.join(current_path, "models")
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        torch.save(agent.actor.state_dict(), os.path.join(save_dir, f"sac_actor_{timestamp}.pth"))
        print(f"saving model with best reward: {best_reward:.1f}")

    print(f"Episode {episode}, Reward: {episode_reward:.1f} Avg_Reward(10): {avg_reward:.1f}")

env.close()

plt.plot(np.arange(len(REWARD_BUFFER)), REWARD_BUFFER, label='Reward')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("SAC on LunarLander")
plt.legend()
plt.show()
