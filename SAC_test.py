import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ===========================
# SAC Actor（与你训练时一致）
# ===========================

class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
        return mu, std

    def sample_normal(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        raw_action = dist.sample()
        action = torch.tanh(raw_action) * self.max_action
        return action


# ===========================
# 测试函数（SAC 专用）
# ===========================

def test_sac(env, actor, episodes=10):
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(200):
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            # SAC 必须用 sample_normal
            action = actor.sample_normal(s)
            action = action.cpu().detach().numpy()[0]

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        print(f"Episode {ep}, Reward: {total_reward:.2f}")


# ===========================
# 主程序
# ===========================

if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 修改这里：你的 SAC 模型路径
    current_path = os.path.dirname(os.path.realpath(__file__))
    model = current_path + "/models/"
    actor_path = model + "sac_actor_20260317182723.pth"

    # 创建 SAC Actor
    actor = SACActor(state_dim, action_dim, max_action).to(device)

    # 加载模型参数
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    # 测试
    test_sac(env, actor, episodes=10)
