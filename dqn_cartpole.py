import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import Gridworld
from gymnasium.wrappers import FlattenObservation
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:{device}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Q网络定义
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x): # 前向传播函数
        return self.fc(x)
    
def train_DQN():
    set_seed(42)
    # 环境 & 参数
    env = gym.make("gymnasium_env/GridWorld-v0")
    env = FlattenObservation(env) # dic->向量
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    qnet = QNet(state_dim, action_dim).to(device)
    qnet_target = QNet(state_dim, action_dim).to(device)
    qnet_target.load_state_dict(qnet.state_dict())

    optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Replay Buffer
    buffer = deque(maxlen=10000)
    gamma = 0.99
    epsilon = 0.2  # 固定探索率，保持模型鲁棒性

    def select_action(state):
        if random.random() < epsilon:  # 20% 随机探索，80% 贪心
            return env.action_space.sample()
        with torch.no_grad():
            return qnet(torch.tensor(state, dtype=torch.float32, device=device)).argmax().item()
        
    rewards = []
    # 训练循环
    best_reward = -np.inf
    for episode in range(1200):
        state, _ = env.reset()
        total_reward = 0
        
        for t in range(300):
            action = select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # 更新网络
            if len(buffer) > 64:
                batch = random.sample(buffer, 64)
                s,a,r,s2,d = zip(*batch)

                s = torch.tensor(np.array(s), dtype=torch.float32, device=device)
                a = torch.tensor(np.array(a), dtype=torch.int64, device=device)
                r = torch.tensor(np.array(r), dtype=torch.float32, device=device)
                s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=device)
                d = torch.tensor(np.array(d), dtype=torch.float32, device=device)

                q_values = qnet(s).gather(1, a.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    target = r + gamma * (1-d) * qnet_target(s2).max(1)[0]
                loss = loss_fn(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done: break
        rewards.append(total_reward)

        avg_reward = np.mean(rewards[-10:])  # 最近 10 回合平均

        print(f"Episode {episode}, Reward: {total_reward}")

        # 每隔一定周期同步 target network（硬更新）
        if episode % 5 == 0:
            qnet_target.load_state_dict(qnet.state_dict())

        # 只保存正常表现的模型，过滤异常值
        if episode >= 50 and avg_reward > best_reward and avg_reward > -5:
            best_reward = avg_reward
            current_path = os.path.dirname(os.path.realpath(__file__))
            save_dir = os.path.join(current_path, "models")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d%H%M%S")
            torch.save(qnet.state_dict(), os.path.join(save_dir, f"dqn_gridworld{timestamp}.pth"))
            print(f"saving model with best reward: {best_reward:.1f}")

    env.close()

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN on GridWorld')
    plt.show()

if __name__ == "__main__":
    train_DQN()