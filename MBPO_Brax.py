# pyright: reportMissingImports=false
import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def resolve_log_path(log_file: str) -> str:
    if os.path.isdir(log_file):
        return os.path.join(log_file, "mbpo_brax_train.log")
    return log_file


def try_import_jax_brax():
    try:
        import flax.linen as nn
        import jax
        import jax.numpy as jnp
        import optax
        from flax.training import train_state
        from brax import envs

        return nn, jax, jnp, optax, train_state, envs
    except Exception as exc:
        print("Missing JAX/Brax stack.")
        print("Please install dependencies first, for example:")
        print("  pip install --upgrade pip")
        print("  pip install brax flax optax matplotlib")
        print(f"Import error: {exc}")
        return None


@dataclass
class Config:
    # Brax environment config.
    env_name: str = "halfcheetah"
    backend: str = "spring"
    seed: int = 42

    total_steps: int = 30_000_000
    num_envs: int = 512
    # 0 means use all local devices; >0 selects the first N local devices for pmap.
    num_devices: int = 0
    # Optional: shard environment collection across all local JAX devices.
    multi_device: bool = False
    # Optional: parallelize SAC learner (Stage D) across local devices.
    parallel_learner: bool = False
    episode_length: int = 1000

    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    start_steps: int = 10_000
    # Gradient updates per environment step (can be fractional).
    updates_per_step: float = 0.15
    # Hard cap to avoid very slow iterations when num_envs is large.
    max_sac_updates_per_iter: int = 64
    # Prevent entropy temperature from collapsing too far.
    log_alpha_min: float = -8.0
    # Also cap alpha from growing too large.
    log_alpha_max: float = 2.0

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    model_lr: float = 3e-4
    grad_clip_norm: float = 5.0

    alpha_init: float = 0.2

    real_buffer_size: int = 1_000_000
    model_buffer_size: int = 100_000
    # SAC updates start with only real data, then gradually mix in more model data.
    real_ratio: float = 0.85
    model_warmup_steps: int = 200_000
    # After model warmup, decay real_ratio from 1.0 to `real_ratio` over this many env steps.
    real_ratio_ramp_steps: int = 6_000_000

    ensemble_size: int = 8
    model_train_epochs: int = 10
    model_done_loss_weight: float = 0.2
    model_train_freq: int = 500
    model_rollout_freq: int = 1_000
    model_rollout_batch: int = 3_000
    # Rollout horizon starts small and grows toward model_rollout_horizon.
    model_rollout_horizon_min: int = 1
    model_rollout_horizon: int = 2
    model_rollout_ramp_steps: int = 5_000_000 # 模型 rollout horizon 从最小值逐步增加到最大值所需要的“过渡步数
    # 模型生成的虚拟数据质量在训练初期可能较差，因此在前几个更新周期内，保持较短的模型生成轨迹长度，以减少模型误差的累积对学习的影响。
    model_rollout_quality_warmup_updates: int = 8
    # Power > 1 makes rollout horizon growth slower early on.
    model_rollout_growth_power: float = 2.5
    # Keep only the lowest-uncertainty synthetic transitions.
    # Set to 1.0 to disable top-k hard cap and rely on disagreement threshold only.
    rollout_keep_ratio: float = 0.75
    # Optional hard threshold on ensemble disagreement (state_var/obs_dim + reward_var + done_var).
    # When > 0, a progressive schedule is used from start -> end (0 disables threshold).
    max_model_disagreement_start: float = 0.2  # Stricter early filter
    max_model_disagreement: float = 0.5  # Relax later but keep threshold meaningful
    max_model_disagreement_ramp_steps: int = 10_000_000

    eval_every: int = 200_000
    # Number of parallel environments used only during evaluation.
    eval_num_envs: int = 128
    eval_episodes: int = 10
    eval_episode_length: int = 1000
    log_every: int = 10_000
    smooth_window: int = 8
    log_file: str = "logs/mbpo_brax_train.log"
    append_log: bool = False


class ReplayBuffer:
    # Host-side ring buffer storing transitions as NumPy arrays.
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.not_done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        next_obs: np.ndarray,
        not_done: np.ndarray,
    ) -> None:
        # Supports vectorized env inserts and wrap-around writes.
        n = int(obs.shape[0])
        end = self.ptr + n

        rew2 = rew.reshape(-1, 1).astype(np.float32)
        nd2 = not_done.reshape(-1, 1).astype(np.float32)

        if end <= self.capacity:
            self.obs[self.ptr:end] = obs
            self.act[self.ptr:end] = act
            self.rew[self.ptr:end] = rew2
            self.next_obs[self.ptr:end] = next_obs
            self.not_done[self.ptr:end] = nd2
        else:
            first = self.capacity - self.ptr
            second = end - self.capacity
            self.obs[self.ptr:] = obs[:first]
            self.act[self.ptr:] = act[:first]
            self.rew[self.ptr:] = rew2[:first]
            self.next_obs[self.ptr:] = next_obs[:first]
            self.not_done[self.ptr:] = nd2[:first]

            self.obs[:second] = obs[first:]
            self.act[:second] = act[first:]
            self.rew[:second] = rew2[first:]
            self.next_obs[:second] = next_obs[first:]
            self.not_done[:second] = nd2[first:]

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=(batch_size,))
        return {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],
            "next_obs": self.next_obs[idx],
            "not_done": self.not_done[idx],
        }

    def sample_obs(self, batch_size: int) -> np.ndarray:
        idx = np.random.randint(0, self.size, size=(batch_size,))
        return self.obs[idx]

    def all_data(self) -> dict[str, np.ndarray]:
        n = self.size
        return {
            "obs": self.obs[:n],
            "act": self.act[:n],
            "rew": self.rew[:n],
            "next_obs": self.next_obs[:n],
            "not_done": self.not_done[:n],
        }


class RunningNorm:
    # Running statistics used to normalize dynamics model inputs/targets.
    def __init__(self, in_dim: int, out_dim: int):
        self.in_mean = np.zeros((in_dim,), dtype=np.float32)
        self.in_std = np.ones((in_dim,), dtype=np.float32)
        self.out_mean = np.zeros((out_dim,), dtype=np.float32)
        self.out_std = np.ones((out_dim,), dtype=np.float32)

    def update(self, obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray, rew: np.ndarray) -> None:
        sa = np.concatenate([obs, act], axis=-1)
        delta = next_obs - obs
        out = np.concatenate([delta, rew], axis=-1)

        self.in_mean = sa.mean(axis=0)
        self.in_std = np.clip(sa.std(axis=0), 1e-3, None)
        self.out_mean = out.mean(axis=0)
        self.out_std = np.clip(out.std(axis=0), 1e-3, None)



def build_modules(nn, jnp, obs_dim: int, act_dim: int, hidden_dim: int):
    class Actor(nn.Module):
        @nn.compact
        def __call__(self, obs):
            x = nn.Dense(hidden_dim)(obs)
            x = nn.relu(x)
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
            mu = nn.Dense(act_dim)(x)
            log_std = nn.Dense(act_dim)(x)
            log_std = jnp.clip(log_std, -5.0, 2.0)
            return mu, log_std

    class Critic(nn.Module):
        @nn.compact
        def __call__(self, obs, act):
            x = jnp.concatenate([obs, act], axis=-1)
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
            q = nn.Dense(1)(x)
            return q.squeeze(-1)

    class Dynamics(nn.Module):
        @nn.compact
        def __call__(self, obs, act):
            x = jnp.concatenate([obs, act], axis=-1)
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
            # Predict [delta_state, reward] with heteroscedastic uncertainty.
            out_dim = obs_dim + 1
            mean = nn.Dense(out_dim)(x)
            logvar = nn.Dense(out_dim)(x)
            logvar = jnp.clip(logvar, -10.0, 2.0)
            # A separate termination head predicts done probability.
            done_logit = nn.Dense(1)(x)
            return mean, logvar, done_logit

    return Actor(), Critic(), Dynamics()


def make_sac_fns(
    jax,
    jnp,
    optax,
    actor_def,
    critic_def,
    max_action: float,
    gamma: float,
    target_entropy: float,
    alpha_tx,
    log_alpha_min: float,
    log_alpha_max: float,
    axis_name: str | None = None,
):
    log_2pi = jnp.log(2.0 * jnp.pi).astype(jnp.float32)

    def sample_action(actor_params, obs, key, deterministic: bool):
        # Tanh-Gaussian policy: sample pre_tanh action then squash to [-1, 1].
        mu, log_std = actor_def.apply(actor_params, obs) # 前向计算，类似于pytorch的forward函数，输入是actor_params和obs，输出是mu和log_std
        std = jnp.exp(log_std)
        if deterministic:
            pre_tanh = mu
        else: # 重参数化采样
            noise = jax.random.normal(key, shape=mu.shape)
            pre_tanh = mu + std * noise
        squashed = jnp.tanh(pre_tanh)
        action = squashed * max_action

        gaussian_logp = -0.5 * (((pre_tanh - mu) / (std + 1e-8)) ** 2 + 2.0 * log_std + log_2pi)
        gaussian_logp = gaussian_logp.sum(axis=-1)
        correction = jnp.log(1.0 - squashed**2 + 1e-6).sum(axis=-1)
        logp = gaussian_logp - correction
        return action, logp

    @jax.jit
    def update_step( # 一整套 SAC 一步更新，顺序是“先 critic，再 actor，再 alpha，再 target网络软更新”
        actor_state,
        critic1_state,
        critic2_state,
        target_critic1_params,
        target_critic2_params,
        log_alpha,
        alpha_opt_state,
        batch,
        tau: float,
        key,
    ):
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"].squeeze(-1)
        next_obs = batch["next_obs"]
        not_done = batch["not_done"].squeeze(-1)

        # 随机种子key1 for bootstrap target action, key2 for policy gradient action.
        key1, key2 = jax.random.split(key)

        alpha = jnp.exp(log_alpha)

        next_action, next_logp = sample_action(actor_state.params, next_obs, key1, deterministic=False)
        next_q1 = critic_def.apply(target_critic1_params, next_obs, next_action)
        next_q2 = critic_def.apply(target_critic2_params, next_obs, next_action)
        # SAC Bellman backup with entropy bonus.
        target_q = rew + gamma * not_done * (jnp.minimum(next_q1, next_q2) - alpha * next_logp)

        def critic_loss_fn(critic1_params, critic2_params):
            q1 = critic_def.apply(critic1_params, obs, act)
            q2 = critic_def.apply(critic2_params, obs, act)
            loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return loss

        critic_loss, (critic1_grads, critic2_grads) = jax.value_and_grad(
            critic_loss_fn, argnums=(0, 1)
        )(critic1_state.params, critic2_state.params) # 计算critic_loss，并且分别计算critic1_params和critic2_params的梯度

        if axis_name is not None: # 在多设备并行学习模式下，使用jax.lax.pmean在设备间平均梯度和损失，以保持同步更新。
            critic1_grads = jax.lax.pmean(critic1_grads, axis_name=axis_name)
            critic2_grads = jax.lax.pmean(critic2_grads, axis_name=axis_name)
            critic_loss = jax.lax.pmean(critic_loss, axis_name=axis_name)
        # 应用梯度更新critic网络参数，得到新的critic_state。
        # apply_gradients是Flax中TrainState的方法，接受计算得到的梯度并返回一个新的TrainState，其中参数已经更新。
        # 类似于PyTorch中的optimizer.step()，但在Flax中是函数式的，返回新的状态而不是原地修改。
        critic1_state = critic1_state.apply_gradients(grads=critic1_grads)
        critic2_state = critic2_state.apply_gradients(grads=critic2_grads)

        def actor_loss_fn(actor_params):
            new_action, logp = sample_action(actor_params, obs, key2, deterministic=False)
            q1_pi = critic_def.apply(critic1_state.params, obs, new_action)
            q2_pi = critic_def.apply(critic2_state.params, obs, new_action)
            q_pi = jnp.minimum(q1_pi, q2_pi)
            return (alpha * logp - q_pi).mean(), logp

        (actor_loss, logp), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        if axis_name is not None:
            actor_grads = jax.lax.pmean(actor_grads, axis_name=axis_name)
            actor_loss = jax.lax.pmean(actor_loss, axis_name=axis_name)
            logp = jax.lax.pmean(logp, axis_name=axis_name)
        actor_state = actor_state.apply_gradients(grads=actor_grads)

        def alpha_loss_fn(log_alpha_val):
            # Tune alpha to match target entropy.
            entropy_term = jax.lax.stop_gradient(logp + target_entropy)
            return -(log_alpha_val * entropy_term.mean())

        alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(log_alpha)
        if axis_name is not None:
            alpha_grad = jax.lax.pmean(alpha_grad, axis_name=axis_name)
            alpha_loss = jax.lax.pmean(alpha_loss, axis_name=axis_name)
        alpha_updates, alpha_opt_state = alpha_tx.update(alpha_grad, alpha_opt_state)
        log_alpha = optax.apply_updates(log_alpha, alpha_updates)
        log_alpha = jnp.clip(log_alpha, log_alpha_min, log_alpha_max)

        # Polyak averaging for target critics.
        target_critic1_params = jax.tree.map(
            lambda t, s: (1.0 - tau) * t + tau * s,
            target_critic1_params,
            critic1_state.params,
        )
        target_critic2_params = jax.tree.map(
            lambda t, s: (1.0 - tau) * t + tau * s,
            target_critic2_params,
            critic2_state.params,
        )

        metrics = {
            "actor": actor_loss,
            "critic": critic_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(log_alpha),
        }
        return (
            actor_state,
            critic1_state,
            critic2_state,
            target_critic1_params,
            target_critic2_params,
            log_alpha,
            alpha_opt_state,
            metrics,
        )

    return sample_action, update_step


def make_dynamics_fns(jax, jnp, optax, dynamics_def, done_loss_weight: float):
    @jax.jit
    def train_one_model(model_state, batch, in_mean, in_std, out_mean, out_std):
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        not_done = batch["not_done"]

        # MBPO dynamics target: delta state and one-step reward.
        target = jnp.concatenate([next_obs - obs, rew], axis=-1)
        done_target = 1.0 - not_done

        sa = jnp.concatenate([obs, act], axis=-1)
        sa_n = (sa - in_mean) / (in_std + 1e-6)
        tar_n = (target - out_mean) / (out_std + 1e-6)

        s_n = sa_n[:, : obs.shape[-1]]
        a_n = sa_n[:, obs.shape[-1] :]

        def loss_fn(params):
            mu, logvar, done_logit = dynamics_def.apply(params, s_n, a_n)
            inv_var = jnp.exp(-logvar)
            # Gaussian NLL lets each model express aleatoric uncertainty.
            nll = ((mu - tar_n) ** 2 * inv_var + logvar).mean()
            done_bce = optax.sigmoid_binary_cross_entropy(done_logit, done_target).mean()
            return nll + done_loss_weight * done_bce

        loss, grads = jax.value_and_grad(loss_fn)(model_state.params)
        model_state = model_state.apply_gradients(grads=grads)
        return model_state, loss

    @jax.jit
    def predict_one(model_params, obs, act, in_mean, in_std, out_mean, out_std, key):
        noise_key, done_key = jax.random.split(key)
        sa = jnp.concatenate([obs, act], axis=-1)
        sa_n = (sa - in_mean) / (in_std + 1e-6) # 输入归一化，减去均值除以标准差
        obs_dim = obs.shape[-1]
        s_n = sa_n[:, :obs_dim]
        a_n = sa_n[:, obs_dim:]

        mu, logvar, done_logit = dynamics_def.apply(model_params, s_n, a_n)
        std = jnp.exp(0.5 * logvar)
        pred_n = mu + std * jax.random.normal(noise_key, shape=mu.shape)
        pred = pred_n * (out_std + 1e-6) + out_mean
        delta = pred[:, :obs_dim]
        reward = pred[:, obs_dim:]
        next_obs = obs + delta
        done_prob = jax.nn.sigmoid(done_logit)
        done_sample = jax.random.bernoulli(done_key, p=done_prob).astype(jnp.float32)
        not_done_pred = 1.0 - done_sample
        return next_obs, reward, not_done_pred

    @jax.jit
    def predict_ensemble(model_params_stack, obs, act, in_mean, in_std, out_mean, out_std, keys):
        def one_model(params_i, key_i):
            return predict_one(params_i, obs, act, in_mean, in_std, out_mean, out_std, key_i)

        ens_next, ens_rew, ens_not_done = jax.vmap(one_model, in_axes=(0, 0))(model_params_stack, keys)
        return ens_next, ens_rew, ens_not_done

    return train_one_model, predict_one, predict_ensemble


def evaluate_policy(
    cfg: Config,
    actor_params,
    sample_action_fn,
    reset_fn,
    step_fn,
    jax,
    jnp,
) -> dict[str, float]:
    # Fast deterministic evaluation: vectorized envs + scan over time, with done-masking.
    eval_envs = max(1, int(cfg.eval_num_envs))

    @jax.jit
    def rollout_once(eval_key):
        reset_keys = jax.random.split(eval_key, eval_envs)
        init_state = reset_fn(reset_keys)
        init_done = jnp.zeros((eval_envs,), dtype=jnp.bool_)
        init_ret = jnp.zeros((eval_envs,), dtype=jnp.float32)
        init_run = jnp.zeros((eval_envs,), dtype=jnp.float32)
        init_ctrl = jnp.zeros((eval_envs,), dtype=jnp.float32)

        def scan_body(carry, _):
            state, done_mask, ep_ret, ep_run, ep_ctrl, key = carry
            key, eval_key = jax.random.split(key)
            action, _ = sample_action_fn(actor_params, state.obs, eval_key, deterministic=True)
            next_state = step_fn(state, action)

            alive = (~done_mask).astype(jnp.float32)
            reward = next_state.reward.astype(jnp.float32)
            ep_ret = ep_ret + reward * alive

            metrics = getattr(next_state, "metrics", None)
            if metrics is not None:
                run_val = jnp.asarray(metrics.get("reward_run", jnp.zeros_like(reward)), dtype=jnp.float32)
                ctrl_val = jnp.asarray(metrics.get("reward_ctrl", jnp.zeros_like(reward)), dtype=jnp.float32)
            else:
                run_val = jnp.zeros_like(reward)
                ctrl_val = jnp.zeros_like(reward)

            ep_run = ep_run + run_val * alive
            ep_ctrl = ep_ctrl + ctrl_val * alive
            done_mask = jnp.logical_or(done_mask, next_state.done.astype(jnp.bool_))
            return (next_state, done_mask, ep_ret, ep_run, ep_ctrl, key), None
        

        init_key = eval_key
        (final_state, final_done, ep_ret, ep_run, ep_ctrl, _), _ = jax.lax.scan(
            scan_body,
            (init_state, init_done, init_ret, init_run, init_ctrl, init_key),
            xs=None,
            length=cfg.eval_episode_length,
        )
        del final_state, final_done
        return ep_ret, ep_run, ep_ctrl

    rewards_all = []
    run_all = []
    ctrl_all = []
    for ep in range(cfg.eval_episodes):
        key = jax.random.PRNGKey(cfg.seed + 10000 + ep)
        ep_ret, ep_run, ep_ctrl = rollout_once(key)
        rewards_all.append(np.asarray(jax.device_get(ep_ret), dtype=np.float32))
        run_all.append(np.asarray(jax.device_get(ep_run), dtype=np.float32))
        ctrl_all.append(np.asarray(jax.device_get(ep_ctrl), dtype=np.float32))

    rewards = np.concatenate(rewards_all, axis=0) if rewards_all else np.zeros((0,), dtype=np.float32)
    rewards_run = np.concatenate(run_all, axis=0) if run_all else np.zeros((0,), dtype=np.float32)
    rewards_ctrl = np.concatenate(ctrl_all, axis=0) if ctrl_all else np.zeros((0,), dtype=np.float32)
    return {
        "episode_reward": float(np.mean(rewards)),
        "episode_reward_run": float(np.mean(rewards_run)) if rewards_run.size > 0 else 0.0,
        "episode_reward_ctrl": float(np.mean(rewards_ctrl)) if rewards_ctrl.size > 0 else 0.0,
        "episode_reward_std": float(np.std(rewards)) if rewards.size > 0 else 0.0,
    }


def save_actor_params(path: str, actor_params, jax_module) -> None:
    leaves = jax_module.tree_util.tree_leaves(actor_params)
    arrays = {f"arr_{i}": np.asarray(jax_module.device_get(v)) for i, v in enumerate(leaves)}
    np.savez(path, **arrays)


def train(cfg: Config) -> None:
    imported = try_import_jax_brax()
    if imported is None:
        return

    nn, jax, jnp, optax, train_state, envs = imported
    available_devices = jax.local_devices()
    available_local_devices = len(available_devices)
    if cfg.num_devices < 0:
        raise ValueError(f"num_devices must be >= 0, got {cfg.num_devices}")
    if cfg.num_devices == 0:
        selected_devices = available_devices
    else:
        if cfg.num_devices > available_local_devices:
            raise ValueError(
                f"num_devices ({cfg.num_devices}) exceeds available local devices ({available_local_devices})."
            )
        selected_devices = available_devices[: cfg.num_devices]

    local_devices = len(selected_devices)
    global_devices = jax.device_count()
    use_multi_device = bool(cfg.multi_device and local_devices > 1)
    use_parallel_learner = bool(cfg.parallel_learner and use_multi_device)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX local devices available ({available_local_devices}): {available_devices}")
    print(f"JAX selected local devices ({local_devices}): {selected_devices}")
    print(f"JAX global device_count: {global_devices} | process_index: {jax.process_index()}")

    if cfg.multi_device and local_devices <= 1:
        print("multi_device=True but only one local JAX device found; falling back to single-device mode.")
    if cfg.parallel_learner and not use_multi_device:
        print("parallel_learner=True but multi_device collection is not active; falling back to single-device learner.")
    if use_multi_device and (cfg.num_envs % local_devices != 0):
        raise ValueError(
            f"num_envs ({cfg.num_envs}) must be divisible by local_device_count ({local_devices}) in multi-device mode."
        )
    if use_parallel_learner and (cfg.batch_size % local_devices != 0):
        raise ValueError(
            f"batch_size ({cfg.batch_size}) must be divisible by local_device_count ({local_devices}) in parallel learner mode."
        )
    envs_per_device = cfg.num_envs // local_devices if use_multi_device else cfg.num_envs

    print(
        "Config | "
        f"actor_lr={cfg.actor_lr:.2e} critic_lr={cfg.critic_lr:.2e} "
        f"alpha_lr={cfg.alpha_lr:.2e} model_lr={cfg.model_lr:.2e} "
        f"updates_per_step={cfg.updates_per_step:.3f} max_updates={cfg.max_sac_updates_per_iter} "
        f"multi_device={use_multi_device} parallel_learner={use_parallel_learner} "
        f"local_devices={local_devices} global_devices={global_devices} num_devices_arg={cfg.num_devices}"
    )

    np.random.seed(cfg.seed)

    env = envs.get_environment(env_name=cfg.env_name, backend=cfg.backend)

    # Vectorized Brax stepping for parallel data collection.
    # In multi-device mode, we shard envs across devices with pmap(vmap).
    eval_reset_fn = jax.jit(jax.vmap(env.reset))
    eval_step_fn = jax.jit(jax.vmap(env.step))
    if use_multi_device:
        reset_fn = jax.jit(jax.pmap(jax.vmap(env.reset), devices=selected_devices))
        step_fn = jax.jit(jax.pmap(jax.vmap(env.step), devices=selected_devices))
    else:
        reset_fn = eval_reset_fn
        step_fn = eval_step_fn

    if use_multi_device:
        def step_collect_fn(curr_state, action, reset_keys):
            # Step once and record transition from raw next state.
            next_state_raw = step_fn(curr_state, action)
            done = next_state_raw.done.astype(jnp.bool_)
            not_done = 1.0 - next_state_raw.done.astype(jnp.float32)
            transition = (curr_state.obs, action, next_state_raw.obs, next_state_raw.reward, not_done)

            # Auto-reset done environments so subsequent collection does not stay in terminal states.
            reset_state = reset_fn(reset_keys)

            def merge_state(raw_val, reset_val):
                if not hasattr(raw_val, "shape"):
                    return raw_val
                if raw_val.ndim < done.ndim:
                    return raw_val
                if raw_val.shape[: done.ndim] != done.shape:
                    return raw_val
                mask = done
                while mask.ndim < raw_val.ndim:
                    mask = mask[..., None]
                return jnp.where(mask, reset_val, raw_val)

            next_state = jax.tree.map(merge_state, next_state_raw, reset_state)
            return next_state, transition
    else:
        @jax.jit
        def step_collect_fn(curr_state, action, reset_keys):
            # Step once and record transition from raw next state.
            next_state_raw = step_fn(curr_state, action)
            done = next_state_raw.done.astype(jnp.bool_)
            not_done = 1.0 - next_state_raw.done.astype(jnp.float32)
            transition = (curr_state.obs, action, next_state_raw.obs, next_state_raw.reward, not_done)

            # Auto-reset done environments so subsequent collection does not stay in terminal states.
            reset_state = reset_fn(reset_keys)

            def merge_state(raw_val, reset_val):
                if raw_val.ndim == 0:
                    return raw_val
                mask = done.reshape((done.shape[0],) + (1,) * (raw_val.ndim - 1))
                return jnp.where(mask, reset_val, raw_val)

            next_state = jax.tree.map(merge_state, next_state_raw, reset_state)
            return next_state, transition

    key = jax.random.PRNGKey(cfg.seed)
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, cfg.num_envs)
    if use_multi_device:
        reset_keys = reset_keys.reshape((local_devices, envs_per_device, 2))
    env_state = reset_fn(reset_keys)

    obs0 = np.asarray(jax.device_get(env_state.obs))
    obs_dim = int(obs0.shape[-1])
    act_dim = int(env.action_size)
    max_action = 1.0
    print("obs_dim", obs_dim, "act_dim", act_dim, "num_envs", cfg.num_envs) #debug

    actor_def, critic_def, dynamics_def = build_modules(nn, jnp, obs_dim, act_dim, cfg.hidden_dim)

    init_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    init_act = jnp.zeros((1, act_dim), dtype=jnp.float32)

    key, actor_key, c1_key, c2_key = jax.random.split(key, 4)

    actor_params = actor_def.init(actor_key, init_obs)
    critic1_params = critic_def.init(c1_key, init_obs, init_act)
    critic2_params = critic_def.init(c2_key, init_obs, init_act)
    # 每个网络都使用Adam优化器，并且在更新前进行全局梯度裁剪，以提高训练稳定性。
    actor_tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip_norm), optax.adam(cfg.actor_lr))
    critic_tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip_norm), optax.adam(cfg.critic_lr))
    model_tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip_norm), optax.adam(cfg.model_lr))

    actor_state = train_state.TrainState.create(
        apply_fn=actor_def.apply,
        params=actor_params,
        tx=actor_tx,
    )
    critic1_state = train_state.TrainState.create(
        apply_fn=critic_def.apply,
        params=critic1_params,
        tx=critic_tx,
    )
    critic2_state = train_state.TrainState.create(
        apply_fn=critic_def.apply,
        params=critic2_params,
        tx=critic_tx,
    )

    # Target networks are initialized from online critics.
    target_critic1_params = critic1_state.params
    target_critic2_params = critic2_state.params

    # Optimize log(alpha) for stability and positivity of alpha.
    log_alpha = jnp.array(np.log(cfg.alpha_init), dtype=jnp.float32)
    alpha_tx = optax.adam(cfg.alpha_lr)
    alpha_opt_state = alpha_tx.init(log_alpha)

    # Ensemble of dynamics models to reduce model bias.
    model_states = []
    for i in range(cfg.ensemble_size):
        key, mkey = jax.random.split(key)
        m_params = dynamics_def.init(mkey, init_obs, init_act)
        m_state = train_state.TrainState.create(
            apply_fn=dynamics_def.apply,
            params=m_params,
            tx=model_tx,
        )
        model_states.append(m_state)

    sample_action_fn, sac_update_fn = make_sac_fns(
        jax,
        jnp,
        optax,
        actor_def,
        critic_def,
        max_action=max_action,
        gamma=cfg.gamma,
        target_entropy=-float(act_dim),
        alpha_tx=alpha_tx,
        log_alpha_min=cfg.log_alpha_min,
        log_alpha_max=cfg.log_alpha_max,
    )

    if use_parallel_learner:
        _, sac_update_parallel_base_fn = make_sac_fns(
            jax,
            jnp,
            optax,
            actor_def,
            critic_def,
            max_action=max_action,
            gamma=cfg.gamma,
            target_entropy=-float(act_dim),
            alpha_tx=alpha_tx,
            log_alpha_min=cfg.log_alpha_min,
            log_alpha_max=cfg.log_alpha_max,
            axis_name="devices",
        )
        # 指定输入参数在设备维度上的分片方式
        sac_update_parallel_fn = jax.pmap(
            sac_update_parallel_base_fn,
            axis_name="devices",
            devices=selected_devices,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, 0),
        )

        def sac_multi_update_parallel_base_fn(
            actor_state,
            critic1_state,
            critic2_state,
            target_critic1_params,
            target_critic2_params,
            log_alpha,
            alpha_opt_state,
            obs_batch,
            act_batch,
            rew_batch,
            next_obs_batch,
            not_done_batch,
            tau,
            key_batch,
        ):
            # Run multiple updates per device inside one scan to reduce host-side pmap dispatch overhead.
            def body(carry, xs):
                (
                    a_state,
                    c1_state,
                    c2_state,
                    tc1_params,
                    tc2_params,
                    la,
                    a_opt_state,
                ) = carry
                obs, act, rew, next_obs, not_done, k = xs
                batch = {
                    "obs": obs,
                    "act": act,
                    "rew": rew,
                    "next_obs": next_obs,
                    "not_done": not_done,
                }
                (
                    a_state,
                    c1_state,
                    c2_state,
                    tc1_params,
                    tc2_params,
                    la,
                    a_opt_state,
                    metrics,
                ) = sac_update_parallel_base_fn(
                    a_state,
                    c1_state,
                    c2_state,
                    tc1_params,
                    tc2_params,
                    la,
                    a_opt_state,
                    batch,
                    tau,
                    k,
                )
                new_carry = (a_state, c1_state, c2_state, tc1_params, tc2_params, la, a_opt_state)
                return new_carry, metrics

            init_carry = (
                actor_state,
                critic1_state,
                critic2_state,
                target_critic1_params,
                target_critic2_params,
                log_alpha,
                alpha_opt_state,
            )
            final_carry, metrics_seq = jax.lax.scan(
                body,
                init_carry,
                (obs_batch, act_batch, rew_batch, next_obs_batch, not_done_batch, key_batch),
            )
            last_metrics = jax.tree.map(lambda arr: arr[-1], metrics_seq)
            return (*final_carry, last_metrics)

        sac_multi_update_parallel_fn = jax.pmap(
            sac_multi_update_parallel_base_fn,
            axis_name="devices",
            devices=selected_devices,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 0),
        )

    @jax.jit
    def sac_multi_update_fn(
        actor_state,
        critic1_state,
        critic2_state,
        target_critic1_params,
        target_critic2_params,
        log_alpha,
        alpha_opt_state,
        obs_batch,
        act_batch,
        rew_batch,
        next_obs_batch,
        not_done_batch,
        key_batch,
        tau,
    ):
        # Run multiple SAC updates in one JAX scan to reduce Python dispatch overhead.
        def body(carry, xs):
            (
                a_state,
                c1_state,
                c2_state,
                tc1_params,
                tc2_params,
                la,
                a_opt_state,
            ) = carry
            obs, act, rew, next_obs, not_done, k = xs
            batch = {
                "obs": obs,
                "act": act,
                "rew": rew,
                "next_obs": next_obs,
                "not_done": not_done,
            }
            (
                a_state,
                c1_state,
                c2_state,
                tc1_params,
                tc2_params,
                la,
                a_opt_state,
                metrics,
            ) = sac_update_fn(
                a_state,
                c1_state,
                c2_state,
                tc1_params,
                tc2_params,
                la,
                a_opt_state,
                batch,
                tau,
                k,
            )
            new_carry = (a_state, c1_state, c2_state, tc1_params, tc2_params, la, a_opt_state)
            return new_carry, metrics

        init_carry = (
            actor_state,
            critic1_state,
            critic2_state,
            target_critic1_params,
            target_critic2_params,
            log_alpha,
            alpha_opt_state,
        )
        final_carry, metrics_seq = jax.lax.scan(
            body,
            init_carry,
            (obs_batch, act_batch, rew_batch, next_obs_batch, not_done_batch, key_batch),
        )
        last_metrics = jax.tree.map(lambda arr: arr[-1], metrics_seq)
        return (*final_carry, last_metrics)

    train_model_fn, predict_model_fn, predict_ensemble_fn = make_dynamics_fns(
        jax,
        jnp,
        optax,
        dynamics_def,
        done_loss_weight=cfg.model_done_loss_weight,
    )

    @jax.jit
    def train_ensemble_models_fn(model_states_stacked, batch, in_mean, in_std, out_mean, out_std):
        def one_model_update(model_state, obs, act, rew, next_obs, not_done):
            model_batch = {
                "obs": obs,
                "act": act,
                "rew": rew,
                "next_obs": next_obs,
                "not_done": not_done,
            }
            return train_model_fn(model_state, model_batch, in_mean, in_std, out_mean, out_std)

        new_states, losses = jax.vmap(one_model_update, in_axes=(0, 0, 0, 0, 0, 0))(
            model_states_stacked,
            batch["obs"],
            batch["act"],
            batch["rew"],
            batch["next_obs"],
            batch["not_done"],
        )
        return new_states, losses

    def stack_model_states(states):
        return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)

    def unstack_model_states(stacked_states):
        return [jax.tree.map(lambda x, i=i: x[i], stacked_states) for i in range(cfg.ensemble_size)]

    if use_parallel_learner:
        device_list = selected_devices

        def replicate_tree(x):
            return jax.device_put_replicated(x, device_list)

        actor_state = replicate_tree(actor_state)
        critic1_state = replicate_tree(critic1_state)
        critic2_state = replicate_tree(critic2_state)
        target_critic1_params = replicate_tree(target_critic1_params)
        target_critic2_params = replicate_tree(target_critic2_params)
        log_alpha = replicate_tree(log_alpha)
        alpha_opt_state = replicate_tree(alpha_opt_state)
    # 在并行学习模式下，定义一个函数single_replica来从每个参数树中提取第一个副本，以便在需要单设备参数时使用。
    def single_replica(tree):
        if use_parallel_learner:
            return jax.tree.map(lambda x: x[0], tree)
        return tree

    def current_actor_params():
        return single_replica(actor_state.params)

    def current_alpha_value() -> float:
        if use_parallel_learner:
            la0 = jax.device_get(log_alpha[0])
            return float(np.exp(float(la0)))
        return float(np.exp(float(log_alpha)))

    real_buffer = ReplayBuffer(cfg.real_buffer_size, obs_dim, act_dim)
    model_buffer = ReplayBuffer(cfg.model_buffer_size, obs_dim, act_dim)
    norm = RunningNorm(obs_dim + act_dim, obs_dim + 1)

    total_steps = 0
    update_budget = 0.0
    last_log = 0
    last_log_steps = 0
    last_log_time = time.time()
    last_eval = 0
    best_eval = -1e18
    best_step = 0
    model_loss_ema = None
    model_loss_baseline = None
    model_loss_updates = 0
    rollout_disagreement_ema = None
    rollout_disagreement_baseline = None
    rollout_disagreement_updates = 0
    os.makedirs("models", exist_ok=True)
    best_actor_path = os.path.join("models", "mbpo_brax_best_actor.npz")
    last_actor_path = os.path.join("models", "mbpo_brax_last_actor.npz")
    history_steps = []
    history_rewards = []
    history_rewards_run = []
    history_rewards_ctrl = []
    history_rewards_std = []
    # Use elapsed-step triggers so frequency is stable even when num_envs >= freq.
    last_model_train_step = -int(cfg.model_train_freq)
    last_model_rollout_step = -int(cfg.model_rollout_freq)

    def current_rollout_horizon() -> int:
        min_h = max(1, int(cfg.model_rollout_horizon_min))
        max_h = max(min_h, int(cfg.model_rollout_horizon))
        if total_steps < cfg.model_warmup_steps:
            return min_h
        if (
            model_loss_updates < cfg.model_rollout_quality_warmup_updates
            or rollout_disagreement_updates < cfg.model_rollout_quality_warmup_updates
        ):
            return min_h

        ramp = max(1, int(cfg.model_rollout_ramp_steps))
        step_progress = float(np.clip((total_steps - cfg.model_warmup_steps) / float(ramp), 0.0, 1.0))
        step_progress = float(np.clip(step_progress ** float(cfg.model_rollout_growth_power), 0.0, 1.0))
        step_h = min_h + int(round(step_progress * float(max_h - min_h)))

        quality_terms = []
        if model_loss_ema is not None and model_loss_baseline is not None:
            # Use relative improvement so negative losses do not break the scale.
            loss_scale = max(abs(float(model_loss_baseline)), 1e-6)
            loss_score = (float(model_loss_baseline) - float(model_loss_ema)) / loss_scale
            quality_terms.append(float(np.clip(loss_score, 0.0, 1.0)))
        if rollout_disagreement_ema is not None and rollout_disagreement_baseline is not None:
            dis_scale = max(abs(float(rollout_disagreement_baseline)), 1e-6)
            dis_score = (float(rollout_disagreement_baseline) - float(rollout_disagreement_ema)) / dis_scale
            quality_terms.append(float(np.clip(dis_score, 0.0, 1.0)))

        if quality_terms:
            quality = float(np.mean(quality_terms))
            quality_h = min_h + int(round(quality * float(max_h - min_h)))
            return max(min_h, min(max_h, min(step_h, quality_h)))
        return step_h

    def current_real_ratio() -> float:
        if total_steps < cfg.model_warmup_steps:
            return 1.0

        start_ratio = 1.0
        end_ratio = float(np.clip(cfg.real_ratio, 0.0, 1.0))
        if end_ratio >= start_ratio:
            return start_ratio

        ramp = max(1, int(cfg.real_ratio_ramp_steps))
        progress = float(np.clip((total_steps - cfg.model_warmup_steps) / float(ramp), 0.0, 1.0))
        return float(start_ratio + progress * (end_ratio - start_ratio))

    def current_max_model_disagreement() -> float:
        end_thr = float(cfg.max_model_disagreement)
        if end_thr <= 0.0:
            return 0.0

        start_thr = float(cfg.max_model_disagreement_start)
        if start_thr <= 0.0:
            start_thr = end_thr

        # Keep strict threshold before model warmup, then relax toward the configured end threshold.
        if total_steps < cfg.model_warmup_steps:
            return start_thr

        if end_thr <= start_thr:
            return start_thr

        ramp = max(1, int(cfg.max_model_disagreement_ramp_steps))
        progress = float(np.clip((total_steps - cfg.model_warmup_steps) / float(ramp), 0.0, 1.0))
        return float(start_thr + progress * (end_thr - start_thr))

    while total_steps < cfg.total_steps:
        num_updates = 0
        effective_real_ratio = current_real_ratio()
        effective_max_disagreement = current_max_model_disagreement()
        rollout_keep_frac: float | None = None
        threshold_keep_frac: float | None = None
        rollout_horizon_now = current_rollout_horizon()
        # Stage A: collect real transitions from Brax env.
        if total_steps < cfg.start_steps:
            if use_multi_device:
                act_shape = (local_devices, envs_per_device, act_dim)
            else:
                act_shape = (cfg.num_envs, act_dim)
            act_jnp = jnp.asarray(np.random.uniform(-1.0, 1.0, size=act_shape).astype(np.float32))
        else:
            key, akey = jax.random.split(key)
            act_jnp, _ = sample_action_fn(current_actor_params(), env_state.obs, akey, deterministic=False)

        if use_multi_device and act_jnp.ndim == 2:
            act_jnp = act_jnp.reshape((local_devices, envs_per_device, act_dim))

        key, step_reset_key = jax.random.split(key)
        step_reset_keys = jax.random.split(step_reset_key, cfg.num_envs)
        if use_multi_device:
            step_reset_keys = step_reset_keys.reshape((local_devices, envs_per_device, 2))
            expected_prefix = (local_devices, envs_per_device)
            obs_prefix = tuple(env_state.obs.shape[:2])
            act_prefix = tuple(act_jnp.shape[:2])
            key_prefix = tuple(step_reset_keys.shape[:2])
            if obs_prefix != expected_prefix or act_prefix != expected_prefix or key_prefix != expected_prefix:
                raise ValueError(
                    "Multi-device batch shape mismatch before step_collect_fn: "
                    f"expected={expected_prefix}, obs={obs_prefix}, act={act_prefix}, reset_keys={key_prefix}"
                )
        env_state, transition = step_collect_fn(env_state, act_jnp, step_reset_keys)
        obs_np, act_np, next_obs_np, rew_np, not_done_np = jax.device_get(transition)
        obs_np = np.asarray(obs_np, dtype=np.float32).reshape((-1, obs_dim))
        act_np = np.asarray(act_np, dtype=np.float32).reshape((-1, act_dim))
        next_obs_np = np.asarray(next_obs_np, dtype=np.float32).reshape((-1, obs_dim))
        rew_np = np.asarray(rew_np, dtype=np.float32).reshape((-1,))
        not_done_np = np.asarray(not_done_np, dtype=np.float32).reshape((-1,))

        real_buffer.add_batch(obs_np, act_np, rew_np, next_obs_np, not_done_np)
        total_steps += cfg.num_envs

        model_loss_value = 0.0

        if total_steps >= cfg.start_steps and real_buffer.size >= cfg.batch_size:
            # Stage B: periodically refit dynamics ensemble on real data.
            if (total_steps - last_model_train_step) >= int(cfg.model_train_freq):
                last_model_train_step = total_steps
                data = real_buffer.all_data()
                norm.update(data["obs"], data["act"], data["next_obs"], data["rew"])

                in_mean = jnp.asarray(norm.in_mean)
                in_std = jnp.asarray(norm.in_std)
                out_mean = jnp.asarray(norm.out_mean)
                out_std = jnp.asarray(norm.out_std)

                losses = []
                model_states_stacked = stack_model_states(model_states)
                for _ in range(cfg.model_train_epochs):
                    batch_np = real_buffer.sample(cfg.batch_size * cfg.ensemble_size)
                    batch_jnp = {
                        "obs": jnp.asarray(batch_np["obs"]).reshape(cfg.ensemble_size, cfg.batch_size, obs_dim),
                        "act": jnp.asarray(batch_np["act"]).reshape(cfg.ensemble_size, cfg.batch_size, act_dim),
                        "rew": jnp.asarray(batch_np["rew"]).reshape(cfg.ensemble_size, cfg.batch_size, 1),
                        "next_obs": jnp.asarray(batch_np["next_obs"]).reshape(
                            cfg.ensemble_size, cfg.batch_size, obs_dim
                        ),
                        "not_done": jnp.asarray(batch_np["not_done"]).reshape(
                            cfg.ensemble_size, cfg.batch_size, 1
                        ),
                    }
                    model_states_stacked, mloss_vec = train_ensemble_models_fn(
                        model_states_stacked,
                        batch_jnp,
                        in_mean,
                        in_std,
                        out_mean,
                        out_std,
                    )
                    losses.append(float(np.mean(np.asarray(jax.device_get(mloss_vec), dtype=np.float32))))

                model_states = unstack_model_states(model_states_stacked)
                if losses:
                    model_loss_value = float(np.mean(np.asarray(losses, dtype=np.float32)))
                    model_loss_ema = (
                        model_loss_value if model_loss_ema is None else 0.9 * model_loss_ema + 0.1 * model_loss_value
                    )
                    if model_loss_baseline is None:
                        model_loss_baseline = model_loss_ema
                    model_loss_updates += 1

            if (
                (total_steps - last_model_rollout_step) >= int(cfg.model_rollout_freq)
                and real_buffer.size >= cfg.model_rollout_batch
            ):
                last_model_rollout_step = total_steps
                # Stage C: short-horizon synthetic rollouts from real states.
                rollout_obs_jnp = jnp.asarray(real_buffer.sample_obs(cfg.model_rollout_batch))

                in_mean = jnp.asarray(norm.in_mean)
                in_std = jnp.asarray(norm.in_std)
                out_mean = jnp.asarray(norm.out_mean)
                out_std = jnp.asarray(norm.out_std)
                model_params_stack = jax.tree.map(
                    lambda *xs: jnp.stack(xs, axis=0),
                    *[ms.params for ms in model_states],
                )

                for _ in range(rollout_horizon_now):
                    key, rkey = jax.random.split(key)
                    act_jnp, _ = sample_action_fn(current_actor_params(), rollout_obs_jnp, rkey, deterministic=False)

                    # Predict all ensemble outputs in one vmapped call.
                    key, pkeys = jax.random.split(key)
                    ens_keys = jax.random.split(pkeys, cfg.ensemble_size)
                    ens_next, ens_rew, ens_not_done = predict_ensemble_fn(
                        model_params_stack,
                        rollout_obs_jnp,
                        act_jnp,
                        in_mean,
                        in_std,
                        out_mean,
                        out_std,
                        ens_keys,
                    )

                    key, idx_key = jax.random.split(key)
                    model_idx = jax.random.randint(
                        idx_key,
                        (cfg.model_rollout_batch,),
                        0,
                        cfg.ensemble_size,
                    )
                    batch_idx = jnp.arange(cfg.model_rollout_batch)
                    next_obs_pred_jnp = ens_next[model_idx, batch_idx, :]
                    rew_pred_jnp = ens_rew[model_idx, batch_idx, :]
                    not_done_pred_jnp = ens_not_done[model_idx, batch_idx, :]
                    # Normalize disagreement by obs_dim so that state/reward/done parts have comparable scale.
                    disagreement_jnp = (
                        jnp.var(ens_next, axis=0).mean(axis=-1) / float(obs_dim)
                        + jnp.var(ens_rew, axis=0).squeeze(-1)
                        + jnp.var(ens_not_done, axis=0).squeeze(-1)
                    )

                    rollout_obs_np, act_np, next_obs_pred, rew_pred, not_done_pred, disagreement = jax.device_get(
                        (
                            rollout_obs_jnp,
                            act_jnp,
                            next_obs_pred_jnp,
                            rew_pred_jnp,
                            not_done_pred_jnp,
                            disagreement_jnp,
                        )
                    )
                    rollout_obs_np = np.asarray(rollout_obs_np, dtype=np.float32)
                    act_np = np.asarray(act_np, dtype=np.float32)
                    next_obs_pred = np.asarray(next_obs_pred, dtype=np.float32)
                    rew_pred = np.asarray(rew_pred, dtype=np.float32)
                    not_done_pred = np.asarray(not_done_pred, dtype=np.float32).squeeze(-1)
                    disagreement = np.asarray(disagreement, dtype=np.float32)
                    rollout_disagreement_value = float(np.mean(disagreement))
                    rollout_disagreement_ema = (
                        rollout_disagreement_value
                        if rollout_disagreement_ema is None
                        else 0.9 * rollout_disagreement_ema + 0.1 * rollout_disagreement_value
                    )
                    if rollout_disagreement_baseline is None:
                        rollout_disagreement_baseline = rollout_disagreement_ema
                    rollout_disagreement_updates += 1

                    keep_mask = np.ones((cfg.model_rollout_batch,), dtype=bool)
                    threshold_mask = np.ones((cfg.model_rollout_batch,), dtype=bool)
                    if effective_max_disagreement > 0.0:
                        threshold_mask = disagreement <= effective_max_disagreement
                    keep_mask &= threshold_mask
                    threshold_keep_frac = float(np.mean(threshold_mask))

                    if cfg.rollout_keep_ratio < 1.0:
                        keep_k = max(1, int(cfg.model_rollout_batch * cfg.rollout_keep_ratio))
                        low_unc_idx = np.argpartition(disagreement, keep_k - 1)[:keep_k]
                        ratio_mask = np.zeros((cfg.model_rollout_batch,), dtype=bool)
                        ratio_mask[low_unc_idx] = True
                        keep_mask &= ratio_mask

                    kept = int(np.sum(keep_mask))
                    rollout_keep_frac = kept / float(cfg.model_rollout_batch)

                    if kept > 0:
                        not_done_model = np.clip(not_done_pred[keep_mask], 0.0, 1.0).astype(np.float32)
                        model_buffer.add_batch(
                            rollout_obs_np[keep_mask],
                            act_np[keep_mask],
                            rew_pred.squeeze(-1)[keep_mask],
                            next_obs_pred[keep_mask],
                            not_done_model,
                        )
                    rollout_obs_jnp = next_obs_pred_jnp

            # Budgeted updates: accumulate by env steps and cap per iteration.
            update_budget += cfg.updates_per_step * cfg.num_envs
            num_updates = min(int(update_budget), cfg.max_sac_updates_per_iter)
            update_budget -= float(num_updates)
            metrics = {"actor": 0.0, "critic": 0.0, "alpha": current_alpha_value()}

            # Stage D: SAC updates on mixed real/model mini-batches.
            # real_ratio controls how much model data is used each update.
            effective_real_ratio = current_real_ratio()
            real_bs = int(cfg.batch_size * effective_real_ratio)
            model_bs = cfg.batch_size - real_bs
            if real_buffer.size >= real_bs and (model_bs == 0 or model_buffer.size >= model_bs) and num_updates > 0:
                obs_buf = np.empty((num_updates, cfg.batch_size, obs_dim), dtype=np.float32)
                act_buf = np.empty((num_updates, cfg.batch_size, act_dim), dtype=np.float32)
                rew_buf = np.empty((num_updates, cfg.batch_size, 1), dtype=np.float32)
                next_obs_buf = np.empty((num_updates, cfg.batch_size, obs_dim), dtype=np.float32)
                not_done_buf = np.empty((num_updates, cfg.batch_size, 1), dtype=np.float32)

                for i in range(num_updates):
                    rb = real_buffer.sample(real_bs)
                    if model_bs > 0:
                        mb = model_buffer.sample(model_bs)
                        obs_buf[i] = np.concatenate([rb["obs"], mb["obs"]], axis=0)
                        act_buf[i] = np.concatenate([rb["act"], mb["act"]], axis=0)
                        rew_buf[i] = np.concatenate([rb["rew"], mb["rew"]], axis=0)
                        next_obs_buf[i] = np.concatenate([rb["next_obs"], mb["next_obs"]], axis=0)
                        not_done_buf[i] = np.concatenate([rb["not_done"], mb["not_done"]], axis=0)
                    else:
                        obs_buf[i] = rb["obs"]
                        act_buf[i] = rb["act"]
                        rew_buf[i] = rb["rew"]
                        next_obs_buf[i] = rb["next_obs"]
                        not_done_buf[i] = rb["not_done"]

                if use_parallel_learner:
                    per_device_bs = cfg.batch_size // local_devices
                    obs_scan = jnp.asarray(obs_buf).reshape(num_updates, local_devices, per_device_bs, obs_dim)
                    act_scan = jnp.asarray(act_buf).reshape(num_updates, local_devices, per_device_bs, act_dim)
                    rew_scan = jnp.asarray(rew_buf).reshape(num_updates, local_devices, per_device_bs, 1)
                    next_obs_scan = jnp.asarray(next_obs_buf).reshape(num_updates, local_devices, per_device_bs, obs_dim)
                    not_done_scan = jnp.asarray(not_done_buf).reshape(num_updates, local_devices, per_device_bs, 1)

                    # pmap expects device axis first: [devices, updates, per_device_bs, ...].
                    obs_scan = jnp.swapaxes(obs_scan, 0, 1)
                    act_scan = jnp.swapaxes(act_scan, 0, 1)
                    rew_scan = jnp.swapaxes(rew_scan, 0, 1)
                    next_obs_scan = jnp.swapaxes(next_obs_scan, 0, 1)
                    not_done_scan = jnp.swapaxes(not_done_scan, 0, 1)

                    key, update_key = jax.random.split(key)
                    device_scan_keys = jax.random.split(update_key, local_devices * num_updates).reshape(
                        local_devices, num_updates, 2
                    )

                    (
                        actor_state,
                        critic1_state,
                        critic2_state,
                        target_critic1_params,
                        target_critic2_params,
                        log_alpha,
                        alpha_opt_state,
                        m_last,
                    ) = sac_multi_update_parallel_fn(
                        actor_state,
                        critic1_state,
                        critic2_state,
                        target_critic1_params,
                        target_critic2_params,
                        log_alpha,
                        alpha_opt_state,
                        obs_scan,
                        act_scan,
                        rew_scan,
                        next_obs_scan,
                        not_done_scan,
                        cfg.tau,
                        device_scan_keys,
                    )
                    metrics = {
                        "actor": float(jax.device_get(m_last["actor"][0])),
                        "critic": float(jax.device_get(m_last["critic"][0])),
                        "alpha": float(jax.device_get(m_last["alpha"][0])),
                    }
                else:
                    key, scan_key = jax.random.split(key)
                    scan_keys = jax.random.split(scan_key, num_updates)

                    (
                        actor_state,
                        critic1_state,
                        critic2_state,
                        target_critic1_params,
                        target_critic2_params,
                        log_alpha,
                        alpha_opt_state,
                        m,
                    ) = sac_multi_update_fn(
                        actor_state,
                        critic1_state,
                        critic2_state,
                        target_critic1_params,
                        target_critic2_params,
                        log_alpha,
                        alpha_opt_state,
                        jnp.asarray(obs_buf),
                        jnp.asarray(act_buf),
                        jnp.asarray(rew_buf),
                        jnp.asarray(next_obs_buf),
                        jnp.asarray(not_done_buf),
                        scan_keys,
                        cfg.tau,
                    )

                    metrics = {
                        "actor": float(m["actor"]),
                        "critic": float(m["critic"]),
                        "alpha": float(m["alpha"]),
                    }
        else:
            metrics = {"actor": 0.0, "critic": 0.0, "alpha": current_alpha_value()}

        if total_steps - last_log >= cfg.log_every:
            now = time.time()
            step_delta = total_steps - last_log_steps
            time_delta = now - last_log_time
            sps = int(step_delta / max(time_delta, 1e-6))
            last_log = total_steps
            last_log_steps = total_steps
            last_log_time = now
            rollkeep_str = f"{rollout_keep_frac:.2f}" if rollout_keep_frac is not None else "N/A"
            threshkeep_str = f"{threshold_keep_frac:.2f}" if threshold_keep_frac is not None else "N/A"
            dis_ema_str = f"{rollout_disagreement_ema:.2f}" if rollout_disagreement_ema is not None else "N/A"
            print(
                f"Step {total_steps:8d} | SPS {sps:6d} | "
                f"Actor {metrics['actor']:.4f} | Critic {metrics['critic']:.4f} | "
                f"Alpha {metrics['alpha']:.4f} | ModelLoss {model_loss_value:.4f} | "
                f"Upd {num_updates:2d} | RealRatio {effective_real_ratio:.3f} | ModelRatio {1.0 - effective_real_ratio:.3f} | "
                f"MaxDis {effective_max_disagreement:.2f} | MeanDis {dis_ema_str:>5} | "
                f"ThreshKeep {threshkeep_str:>4} | RollKeep {rollkeep_str:>4} | RollH {rollout_horizon_now:2d}"
            )

        if total_steps - last_eval >= cfg.eval_every:
            last_eval = total_steps
            eval_metrics = evaluate_policy(
                cfg,
                current_actor_params(),
                sample_action_fn,
                eval_reset_fn,
                eval_step_fn,
                jax,
                jnp,
            )
            eval_ret = float(eval_metrics["episode_reward"])
            history_steps.append(total_steps)
            history_rewards.append(eval_ret)
            history_rewards_run.append(float(eval_metrics["episode_reward_run"]))
            history_rewards_ctrl.append(float(eval_metrics["episode_reward_ctrl"]))
            history_rewards_std.append(float(eval_metrics["episode_reward_std"]))

            # Always keep latest evaluated checkpoint.
            save_actor_params(last_actor_path, current_actor_params(), jax)

            if eval_ret > best_eval:
                best_eval = eval_ret
                best_step = total_steps
                save_actor_params(best_actor_path, current_actor_params(), jax)
            print(
                f"Eval @ {total_steps:8d} | EpisodeReward {eval_ret:8.2f} | "
                f"Run {eval_metrics['episode_reward_run']:8.2f} | Ctrl {eval_metrics['episode_reward_ctrl']:8.2f} | "
                f"Std {eval_metrics['episode_reward_std']:7.2f} | Best {best_eval:8.2f}"
            )

    if history_steps:
        os.makedirs("plots", exist_ok=True)

        def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
            if window <= 1 or values.size < window:
                return values
            left = window // 2
            right = window - 1 - left
            padded = np.pad(values, (left, right), mode="edge")
            kernel = np.ones((window,), dtype=np.float32) / float(window)
            return np.convolve(padded, kernel, mode="valid")

        rewards = np.array(history_rewards, dtype=np.float32)
        rewards_run = np.array(history_rewards_run, dtype=np.float32)
        rewards_ctrl = np.array(history_rewards_ctrl, dtype=np.float32)
        rewards_std = np.array(history_rewards_std, dtype=np.float32)

        smooth = smooth_1d(rewards, cfg.smooth_window)
        smooth_run = smooth_1d(rewards_run, cfg.smooth_window)
        smooth_ctrl = smooth_1d(rewards_ctrl, cfg.smooth_window)
        smooth_std = smooth_1d(rewards_std, cfg.smooth_window)

        best_idx = int(np.argmax(rewards))
        best_step = history_steps[best_idx]
        best_reward = float(rewards[best_idx])

        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

        axes[0].plot(history_steps, rewards, color="tab:blue", alpha=0.35, label="Episode Reward")
        axes[0].plot(history_steps, smooth, color="tab:orange", label=f"Reward Smoothed({cfg.smooth_window})")
        axes[0].scatter([best_step], [best_reward], color="tab:red", s=35, label=f"Best {best_reward:.1f}")
        axes[0].set_title(f"JAX Brax MBPO Reward Curves ({cfg.env_name})")
        axes[0].set_ylabel("Total Reward")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(history_steps, rewards_run, color="tab:green", alpha=0.35, label="Reward Run")
        axes[1].plot(history_steps, smooth_run, color="tab:olive", label=f"Run Smoothed({cfg.smooth_window})")
        axes[1].plot(history_steps, rewards_ctrl, color="tab:purple", alpha=0.35, label="Reward Ctrl")
        axes[1].plot(history_steps, smooth_ctrl, color="tab:pink", label=f"Ctrl Smoothed({cfg.smooth_window})")
        axes[1].plot(history_steps, smooth_std, color="tab:brown", label=f"Reward Std Smoothed({cfg.smooth_window})")
        axes[1].set_xlabel("Environment Steps")
        axes[1].set_ylabel("Run/Ctrl/Std")
        axes[1].grid(alpha=0.3)
        axes[1].legend(ncol=2)

        out = os.path.join("plots", "mbpo_brax_reward.png")
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved training plot to: {out}")

    # Model selection: prefer best checkpoint for downstream evaluation.
    selected_actor_path = os.path.join("models", "mbpo_brax_actor_selected.npz")
    if os.path.exists(best_actor_path):
        shutil.copyfile(best_actor_path, selected_actor_path)
    elif os.path.exists(last_actor_path):
        shutil.copyfile(last_actor_path, selected_actor_path)

    summary_path = os.path.join("models", "mbpo_brax_selection_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"best_actor={best_actor_path}\n")
        f.write(f"last_actor={last_actor_path}\n")
        f.write(f"selected_actor={selected_actor_path}\n")
        f.write(f"best_eval={best_eval:.6f}\n")
        f.write(f"best_step={best_step}\n")

    print(f"Saved best actor: {best_actor_path}")
    print(f"Saved last actor: {last_actor_path}")
    print(f"Selected actor for evaluation: {selected_actor_path}")
    print(f"Saved selection summary: {summary_path}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="MBPO (JAX + Brax)")
    p.add_argument("--env", type=str, default="halfcheetah")
    p.add_argument("--backend", type=str, default="spring", choices=["spring", "positional", "generalized"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_steps", type=int, default=20_000_000)
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--num_devices", type=int, default=0)
    p.add_argument("--multi_device", action="store_true")
    p.add_argument("--parallel_learner", action="store_true")
    p.add_argument("--episode_length", type=int, default=1000)
    p.add_argument("--start_steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--updates_per_step", type=float, default=0.15)
    p.add_argument("--max_sac_updates_per_iter", type=int, default=64)
    p.add_argument("--log_alpha_min", type=float, default=-8.0)
    p.add_argument("--log_alpha_max", type=float, default=2.0)
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--alpha_lr", type=float, default=3e-4)
    p.add_argument("--model_lr", type=float, default=3e-4)
    p.add_argument("--grad_clip_norm", type=float, default=5.0)
    p.add_argument("--model_done_loss_weight", type=float, default=0.2)
    p.add_argument("--model_train_freq", type=int, default=500)
    p.add_argument("--model_rollout_freq", type=int, default=5000)
    p.add_argument("--model_rollout_horizon", type=int, default=2)
    p.add_argument("--model_rollout_batch", type=int, default=3000)
    p.add_argument("--model_rollout_horizon_min", type=int, default=1)
    p.add_argument("--model_rollout_ramp_steps", type=int, default=5_000_000)
    p.add_argument("--rollout_keep_ratio", type=float, default=0.75)
    p.add_argument("--max_model_disagreement_start", type=float, default=0.2)
    p.add_argument("--max_model_disagreement", type=float, default=0.5)
    p.add_argument("--max_model_disagreement_ramp_steps", type=int, default=10_000_000)
    p.add_argument("--real_ratio", type=float, default=0.85)
    p.add_argument("--model_warmup_steps", type=int, default=200_000)
    p.add_argument("--real_ratio_ramp_steps", type=int, default=6_000_000)
    p.add_argument("--eval_every", type=int, default=200_000)
    p.add_argument("--eval_num_envs", type=int, default=128)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--eval_episode_length", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=10_000)
    p.add_argument("--smooth_window", type=int, default=8)
    p.add_argument("--log_file", type=str, default="logs/mbpo_brax_train.log")
    p.add_argument("--append_log", action="store_true")

    args = p.parse_args()

    cfg = Config()
    cfg.env_name = args.env
    cfg.backend = args.backend
    cfg.seed = args.seed
    cfg.total_steps = args.total_steps
    cfg.num_envs = args.num_envs
    cfg.num_devices = args.num_devices
    cfg.multi_device = args.multi_device
    cfg.parallel_learner = args.parallel_learner
    cfg.episode_length = args.episode_length
    cfg.start_steps = args.start_steps
    cfg.batch_size = args.batch_size
    cfg.updates_per_step = args.updates_per_step
    cfg.max_sac_updates_per_iter = args.max_sac_updates_per_iter
    cfg.log_alpha_min = args.log_alpha_min
    cfg.log_alpha_max = args.log_alpha_max
    cfg.actor_lr = args.actor_lr
    cfg.critic_lr = args.critic_lr
    cfg.alpha_lr = args.alpha_lr
    cfg.model_lr = args.model_lr
    cfg.grad_clip_norm = args.grad_clip_norm
    cfg.model_done_loss_weight = args.model_done_loss_weight
    cfg.model_train_freq = args.model_train_freq
    cfg.model_rollout_freq = args.model_rollout_freq
    cfg.model_rollout_horizon = args.model_rollout_horizon
    cfg.model_rollout_batch = args.model_rollout_batch
    cfg.model_rollout_horizon_min = args.model_rollout_horizon_min
    cfg.model_rollout_ramp_steps = args.model_rollout_ramp_steps
    cfg.rollout_keep_ratio = args.rollout_keep_ratio
    cfg.max_model_disagreement_start = args.max_model_disagreement_start
    cfg.max_model_disagreement = args.max_model_disagreement
    cfg.max_model_disagreement_ramp_steps = args.max_model_disagreement_ramp_steps
    cfg.real_ratio = args.real_ratio
    cfg.model_warmup_steps = args.model_warmup_steps
    cfg.real_ratio_ramp_steps = args.real_ratio_ramp_steps
    cfg.eval_every = args.eval_every
    cfg.eval_num_envs = args.eval_num_envs
    cfg.eval_episodes = args.eval_episodes
    cfg.eval_episode_length = args.eval_episode_length
    cfg.log_every = args.log_every
    cfg.smooth_window = args.smooth_window
    cfg.log_file = args.log_file
    cfg.append_log = args.append_log
    return cfg


if __name__ == "__main__":
    config = parse_args()
    if config.log_file:
        config.log_file = resolve_log_path(config.log_file)
        os.makedirs(os.path.dirname(config.log_file) or ".", exist_ok=True)
        log_mode = "a" if config.append_log else "w"
        with open(config.log_file, log_mode, encoding="utf-8", buffering=1) as log_handle:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = Tee(original_stdout, log_handle)
            sys.stderr = Tee(original_stderr, log_handle)
            try:
                print(f"Logging to: {config.log_file}")
                train(config)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    else:
        train(config)
