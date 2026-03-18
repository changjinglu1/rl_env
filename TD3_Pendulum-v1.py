import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math
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
            nn.Linear(state_dim, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh() # 输出控制到1到-1之间
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, 1) # 输入状态和动作，输出Q值
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, 1) # 输入状态和动作，输出Q值
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1) #(s, a) 组合在一起的向量
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2
    
    def Q1(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.net1(x)
        


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
# 4. DDPG 训练主逻辑
# ======================

def train_td3(
    env_name="Pendulum-v1",
    max_episodes=600,
    max_steps=300,
    gamma=0.99,
    tau=0.005,
    actor_lr=3e-4,
    critic_lr=3e-4,
    batch_size=256,
    start_steps=10000,
    exploration_noise=0.05,
    update_it = 0
):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    actor_target = Actor(state_dim, action_dim, max_action).to(device)
    critic_target = Critic(state_dim, action_dim).to(device)

    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)
    mse_loss = nn.MSELoss()

    replay_buffer = ReplayBuffer()

    total_steps = 0
    returns = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_return = 0

        for step in range(max_steps):
            total_steps += 1

            # 探索：前 start_steps 步随机动作 填满ReplayBuffer
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = actor(s_tensor).cpu().numpy()[0]
                # 加高斯噪声
                action = action + np.random.normal(0, exploration_noise, size=action_dim)
                action = np.clip(action, -max_action, max_action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated

            replay_buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            episode_return += reward

            # 开始更新
            if len(replay_buffer) > batch_size:
                s, a, r, s2, d = replay_buffer.sample(batch_size)

                # Critic 更新
                with torch.no_grad():
                    # 1. target policy smoothing
                    noise = (torch.randn_like(actor_target(s2)) * 0.2).clamp(-0.5,0.5)
                    a2 = (actor_target(s2) + noise).clamp(-max_action, max_action)
                    # 2. 双 Q 取最小
                    target_q1, target_q2 = critic_target(s2, a2)
                    target_q = torch.min(target_q1, target_q2)
                    target_q = r + gamma * (1 - d) * target_q

                current_q1, current_q2 = critic(s, a)
                critic_loss = mse_loss(current_q1, target_q) + mse_loss(current_q2, target_q)

                critic_opt.zero_grad() #清理上一步的久梯度
                critic_loss.backward() #计算 Critic 网络中所有参数的梯度
                critic_opt.step() #根据梯度更新 Critic 的参数 (Adam)

                update_it += 1

                # Actor 更新(最大化 Q(s, a) 等价于最小化 -Q) “延迟更新”
                if update_it % 3 == 0:
                    actor_loss = -critic.Q1(s, actor(s)).mean() # 只用Q1来更新θ

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    # 软更新目标网络
                    soft_update(actor_target, actor, tau)
                    soft_update(critic_target, critic, tau)

            if done:
                break

        returns.append(episode_return)
        print(f"Episode {episode}, Return: {episode_return:.2f}")
        if episode % 50 == 0:
            # save models
            current_path = os.path.dirname(os.path.realpath(__file__))
            save_dir = os.path.join(current_path, "models")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d%H%M%S")
            torch.save(actor.state_dict(), os.path.join(save_dir, f"td3_actor_{timestamp}.pth"))

    env.close()

    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("TD3 on Pendulum-v1")
    plt.show()

    return returns


if __name__ == "__main__":
    train_td3()
