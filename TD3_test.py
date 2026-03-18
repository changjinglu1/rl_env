import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# initialize env
env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
#load model
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + "/models/"
actor_path = model + "ddpg_actor_20260312154833.pth"

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

max_action = float(env.action_space.high[0])
actor = Actor(state_dim, action_dim, max_action).to(device)
actor.load_state_dict(torch.load(actor_path))


for episode in range(20):
    state, _ = env.reset()
    episode_reward = 0
    for step in range(200):
        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = actor(s_tensor).cpu().detach().numpy()[0]

        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        episode_reward += reward
        if terminated or truncated:
            break
    print(f"Episode {episode}, Reward: {episode_reward:.2f}")

env.close()
