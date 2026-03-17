import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ======================
# 1. Actor & Critic 网络
# ======================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)  # 改名避免和张量混淆
        self.max_action = max_action

    def forward(self, s):
        h = self.net(s)
        mu = self.mu(h)
        log_std = self.log_std_head(h).clamp(-20, 2)
        std = log_std.exp()
        return mu, std, log_std

    def sample(self, s):
        mu, std, log_std = self.forward(s)
        eps = torch.randn_like(std)
        pre_tanh = mu + std * eps
        a = torch.tanh(pre_tanh)
        action = self.max_action * a

        # 高斯部分 log π(z|s)
        log_prob = -0.5 * ((pre_tanh - mu) / std).pow(2) - log_std - 0.5 * np.log(2 * np.pi)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # tanh 的 log-det-Jacobian 修正
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2


# ======================
# 2. Replay Buffer
# ======================

class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)

    def add(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        s = torch.tensor(np.array(s), dtype=torch.float32, device=device)
        a = torch.tensor(np.array(a), dtype=torch.float32, device=device)
        r = torch.tensor(np.array(r), dtype=torch.float32, device=device).unsqueeze(1)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=device)
        d = torch.tensor(np.array(d), dtype=torch.float32, device=device).unsqueeze(1)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)


# ======================
# 3. 软更新函数
# ======================

def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


# ======================
# 4. SAC 训练主逻辑
# ======================

def train_sac(
    env_name="Pendulum-v1",
    max_episodes=400,
    max_steps=200,
    gamma=0.99,
    tau=0.005,
    actor_lr=3e-4,
    critic_lr=3e-4,
    batch_size=256,
    start_steps=1000,
    alpha=0.2
):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    critic_target = Critic(state_dim, action_dim).to(device)

    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    replay_buffer = ReplayBuffer()

    total_steps = 0
    returns = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_return = 0

        for step in range(max_steps):
            total_steps += 1

            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action_tensor, _ = actor.sample(s_tensor)
                    action = action_tensor.cpu().numpy()[0]

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            episode_return += reward

            if len(replay_buffer) > batch_size:
                s, a, r, s2, d = replay_buffer.sample(batch_size)

                # ------- 更新 Critic -------
                with torch.no_grad():
                    a2, logp_a2 = actor.sample(s2)
                    target_q1, target_q2 = critic_target(s2, a2)
                    target_q = torch.min(target_q1, target_q2)
                    target_q = r + gamma * (1 - d) * (target_q - alpha * logp_a2)

                current_q1, current_q2 = critic(s, a)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                # ------- 更新 Actor -------
                a_pi, logp_a_pi = actor.sample(s)
                q1_pi, q2_pi = critic(s, a_pi)
                q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (alpha * logp_a_pi - q_pi).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                # ------- 更新 target Critic -------
                soft_update(critic_target, critic, tau)

            if done:
                break

        returns.append(episode_return)
        print(f"Episode {episode}, Return: {episode_return:.2f}")

        if episode % 50 == 0:
            current_path = os.path.dirname(os.path.realpath(__file__))
            save_dir = os.path.join(current_path, "models")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d%H%M%S")
            torch.save(actor.state_dict(), os.path.join(save_dir, f"sac_actor_{timestamp}.pth"))

    env.close()

    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("SAC on Pendulum-v1")
    plt.show()

    return returns


if __name__ == "__main__":
    train_sac()