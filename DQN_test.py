import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import FlattenObservation
from dqn_cartpole import QNet   # 你训练时定义的 QNet
import time
import Gridworld

def dqn_test(env, qnet, episodes):
    for ep in range(episodes):
        state, _ = env.reset()
        env.render()  # 渲染环境
        total_reward = 0
        done = False

        while not done:
            # 选择动作（贪心）
            with torch.no_grad():
                action = qnet(torch.tensor(state, dtype=torch.float32)).argmax().item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            env.render()  # 渲染环境

            total_reward += reward
            state = next_state

            time.sleep(0.2)  # 控制渲染速度

        print("Total Reward:", total_reward)

    env.close()

if __name__ == "__main__":
    # 加载环境
    env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human")
    env = FlattenObservation(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model_path = "models/sac_actor_gridworld20260320144205.pth"   # ← 换成你自己的文件名

    qnet = QNet(state_dim, action_dim)
    qnet.load_state_dict(torch.load(model_path))
    qnet.eval()
    print("Loaded model:", model_path)

    dqn_test(env, qnet, 10)