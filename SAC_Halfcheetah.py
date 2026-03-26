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
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e3, neginf=-1e3)
        log_std = torch.nan_to_num(log_std, nan=0.0, posinf=2.0, neginf=-20.0)
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
    def __init__(self, buffer_size, state_dim, action_dim, device):
        self.device = device
        self.buffer_size = buffer_size
        self.buffer_index = 0
        self.size = 0

        self.state_buffer = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.next_state_buffer = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.action_buffer = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.reward_buffer = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.done_buffer = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        

    def add_buffer(self, state, action, reward, next_state, done, num_envs):
        # 向量化环境：传入的是 numpy batch [num_envs, dim]
        idx = self.buffer_index
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device).reshape(-1, 1)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device).reshape(-1, 1)

        end = idx + num_envs
        if end <= self.buffer_size:
            self.state_buffer[idx:end] = state_t
            self.action_buffer[idx:end] = action_t
            self.reward_buffer[idx:end] = reward_t
            self.next_state_buffer[idx:end] = next_state_t
            self.done_buffer[idx:end] = done_t
        else:
            first_part = self.buffer_size - idx
            second_part = end - self.buffer_size

            self.state_buffer[idx:] = state_t[:first_part]
            self.action_buffer[idx:] = action_t[:first_part]
            self.reward_buffer[idx:] = reward_t[:first_part]
            self.next_state_buffer[idx:] = next_state_t[:first_part]
            self.done_buffer[idx:] = done_t[:first_part]

            self.state_buffer[:second_part] = state_t[first_part:]
            self.action_buffer[:second_part] = action_t[first_part:]
            self.reward_buffer[:second_part] = reward_t[first_part:]
            self.next_state_buffer[:second_part] = next_state_t[first_part:]
            self.done_buffer[:second_part] = done_t[first_part:]

        self.buffer_index = (self.buffer_index + num_envs) % self.buffer_size
        self.size = min(self.size + num_envs, self.buffer_size)

    def sample_buffer(self, batch_size):
        # 零拷贝采样：所有数据已在 GPU，只需 slice        next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
        
        train_dones = terminateds.astype(np.float32)  # 只用 terminated 训练
        dones = np.logical_or(terminateds, truncateds)  # 日志/统计继续用这个
        
        # 对 terminated 的样本，next_state 使用 final_observation（如果有）
        if "final_observation" in infos:
            final_obs = infos["final_observation"]
            for i in range(num_envs):
                if terminateds[i] and final_obs[i] is not None:
                    next_states[i] = final_obs[i]
        
        agent.add_buffer(states, actions, rewards, next_states, train_dones, num_envs)
        index = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.state_buffer[index],
            self.action_buffer[index],
            self.reward_buffer[index],
            self.next_state_buffer[index],
            self.done_buffer[index]
        )


class SACAgent:
    def __init__(self, state_dim, action_dim, buffer_size,
                 alpha, critic_lr, actor_lr, gamma, tau,
                 layer1_dim, layer2_dim, batch_size, max_action):
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size

        # 自动调节 alpha
        self.target_entropy = -action_dim  # SAC 论文推荐
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=actor_lr)
        self.alpha = self.log_alpha.exp().detach()

        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim, device)

        self.critic_1 = CriticNetwork(state_dim, action_dim, layer1_dim, layer2_dim, critic_lr).to(device)
        self.critic_2 = CriticNetwork(state_dim, action_dim, layer1_dim, layer2_dim, critic_lr).to(device)
        self.target_critic_1 = CriticNetwork(state_dim, action_dim, layer1_dim, layer2_dim, critic_lr).to(device)
        self.target_critic_2 = CriticNetwork(state_dim, action_dim, layer1_dim, layer2_dim, critic_lr).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor = ActorNetwork(state_dim, action_dim, layer1_dim, layer2_dim, max_action, actor_lr).to(device)

    def get_actions(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        # 获取随机动作用于探索
        with torch.no_grad():
            actions, _ = self.actor.sample_normal(state_tensor)
        return actions.cpu().numpy()
    
    def add_buffer(self, state, action, reward, next_state, done, num_envs):
        self.buffer.add_buffer(state, action, reward, next_state, done, num_envs)

    # 添加软更新方法
    def update_network_parameters(self):
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self):
        if self.buffer.size < self.batch_size:
            return
        # 零拷贝采样 (所有数据已位于 GPU 且为 Tensor 格式)
        state, action, reward, next_state, done = self.buffer.sample_buffer(self.batch_size)

        state = torch.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        action = torch.nan_to_num(action, nan=0.0, posinf=1e6, neginf=-1e6)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=1e6, neginf=-1e6)
        next_state = torch.nan_to_num(next_state, nan=0.0, posinf=1e6, neginf=-1e6)
        done = torch.nan_to_num(done, nan=1.0, posinf=1.0, neginf=0.0)
        alpha = self.log_alpha.exp().detach()

        # 1. 计算 target Q
        with torch.no_grad():
            next_action, next_logp = self.actor.sample_normal(next_state, reparameterize=False)
            q1_target = self.target_critic_1(next_state, next_action)
            q2_target = self.target_critic_2(next_state, next_action)
            q_target = torch.min(q1_target, q2_target) - alpha * next_logp
            q_target = reward + self.gamma * (1 - done) * q_target
            q_target = torch.nan_to_num(q_target, nan=0.0, posinf=1e6, neginf=-1e6)

        # 2. 更新 Critic
        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        # 【修复 2】加入梯度裁剪，将梯度限制在合理范围，彻底杜绝 NaN
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 1.0)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # 3. 更新 Actor
        new_action, logp = self.actor.sample_normal(state, reparameterize=True)
        q1_pi = self.critic_1(state, new_action)
        q2_pi = self.critic_2(state, new_action)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (alpha * logp - q_pi).mean()

        if not torch.isfinite(critic_loss) or not torch.isfinite(actor_loss):
            return

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        # 【修复 2】加入梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor.optimizer.step()
        self.update_network_parameters()

        # 4. 更新 alpha（自动调节）
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            self.log_alpha.clamp_(min=-20.0, max=2.0)
        self.alpha = self.log_alpha.exp().detach()


if __name__ == "__main__":

    # ======================
    # 3. 训练主循环 (向量版)
    # ======================
    env_name = "HalfCheetah-v5"
    num_envs = 8  # 启动 8 个并行环境
    REPLAYBUFFER_SIZE = 1000000
    max_total_steps = 2_000_000  # 向量化环境看总步数，通常设个 100万 - 200万
    start_steps = 10000 
    batch_size = 256

    # 创建向量化环境并添加 Wrappers
    def make_env():
        def thunk():
            env = gym.make(env_name)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.NormalizeObservation(env)
            # env = gym.wrappers.NormalizeReward(env) 
            return env
        return thunk

    # 先尝试异步向量环境（多进程并行采样），失败时回退到同步版本
    try:
        envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])
        print("Using AsyncVectorEnv")
    except Exception as e:
        print(f"AsyncVectorEnv unavailable, fallback to SyncVectorEnv: {e}")
        envs = gym.vector.SyncVectorEnv([make_env() for _ in range(num_envs)])

    single_state_dim = envs.single_observation_space.shape[0]
    single_action_dim = envs.single_action_space.shape[0]
    max_action = float(envs.single_action_space.high[0])

    agent = SACAgent(single_state_dim, single_action_dim, REPLAYBUFFER_SIZE,
                    alpha=0.2, critic_lr=3e-4, actor_lr=3e-4, gamma=0.99, tau=0.005,
                    layer1_dim=256, layer2_dim=256, batch_size=batch_size,
                    max_action=max_action)

    REWARD_BUFFER = []
    best_reward = -np.inf
    total_steps = 0

    # 向量化环境初始化，拿到 8 个环境的 batch state
    states, _ = envs.reset()

    print("开始训练...")
    start_time = time.time()

    while total_steps < max_total_steps:
        # 1. 动作采样
        if total_steps < start_steps:
            actions = envs.action_space.sample()
        else:
            actions = agent.get_actions(states)

        # 2. 与环境交互
        next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
        dones = np.logical_or(terminateds, truncateds)

        # 3. 存入 Buffer（注意把 bool 转成 float32）
        agent.add_buffer(states, actions, rewards, next_states, dones.astype(np.float32), num_envs)

        states = next_states
        total_steps += num_envs

        # 4. 记录真实的 Episode 奖励 (RecordEpisodeStatistics 自动处理)
        if "episode" in infos:
            # infos["episode"]["r"] 是一个数组，只在对应环境 done 时有真实分数值
            for i, is_done in enumerate(dones):
                if is_done:
                    # 获取该子环境刚结束的这一回合的真实总分
                    ep_reward = infos["episode"]["r"][i]
                    REWARD_BUFFER.append(ep_reward)
                    
                    # 打印日志（每 10 个回合打印一次，避免刷屏）
                    if len(REWARD_BUFFER) % 10 == 0:
                        avg_reward = np.mean(REWARD_BUFFER[-10:])
                        fps = int(total_steps / (time.time() - start_time))
                        print(f"Step: {total_steps}, Env_{i} Reward: {float(ep_reward):.1f}, Avg_Reward(10): {avg_reward:.1f}, FPS: {fps}")

                        # 保存最好模型
                        if avg_reward > best_reward and total_steps > start_steps:
                            best_reward = avg_reward
                            os.makedirs("models", exist_ok=True)
                            torch.save(agent.actor.state_dict(), "models/sac_actor_best.pth")
                            print(f"--> Saved new best model! Reward: {best_reward:.1f}")

        # 5. 网络更新
        if total_steps >= start_steps:
            # 收集了 num_envs 条数据，就连续更新 num_envs 次，拉满 GPU 利用率
            for _ in range(num_envs):
                agent.update()

    envs.close()

    plt.plot(np.arange(len(REWARD_BUFFER)), REWARD_BUFFER, label='Reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("SAC on Halfcheetah")
    plt.legend()
    # plt.show()
    plt.savefig("reward_curve.png")

