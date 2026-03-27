# pyright: reportMissingImports=false
import argparse
import inspect
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast JAX+Brax SAC training (GPU/TPU friendly)")
    parser.add_argument("--env", type=str, default="halfcheetah")
    parser.add_argument("--timesteps", type=int, default=20_000_000)
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--num_eval_envs", type=int, default=128)
    parser.add_argument("--backend", type=str, default="spring", choices=["spring", "positional", "generalized"])
    parser.add_argument("--num_evals", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--entropy_cost", type=float, default=3e-3)
    parser.add_argument("--reward_scaling", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--smooth_window", type=int, default=7)
    args = parser.parse_args()

    try:
        import jax
        import matplotlib.pyplot as plt
        from brax import envs
        from brax.training.agents.sac import train as sac_train
    except Exception as exc:
        print("Missing JAX/Brax stack.")
        print("Please install on your Linux GPU server, e.g.:\n"
              "  pip install --upgrade pip\n"
              "  pip install 'jax[cuda12]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html\n"
              "  pip install brax optax flax matplotlib")
        print(f"Import error: {exc}")
        return

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    env = envs.get_environment(env_name=args.env, backend=args.backend)

    history_steps = []
    history_rewards = []
    history_rewards_run = []
    history_rewards_ctrl = []
    history_rewards_std = []
    start_time = time.time()

    def progress(num_steps: int, metrics: dict) -> None:
        wall = time.time() - start_time
        eval_reward = float(metrics.get("eval/episode_reward", 0.0))
        eval_reward_run = float(metrics.get("eval/episode_reward_run", 0.0))
        eval_reward_ctrl = float(metrics.get("eval/episode_reward_ctrl", 0.0))
        eval_reward_std = float(metrics.get("eval/episode_reward_std", 0.0))
        sps = int(num_steps / max(wall, 1e-6))
        history_steps.append(num_steps)
        history_rewards.append(eval_reward)
        history_rewards_run.append(eval_reward_run)
        history_rewards_ctrl.append(eval_reward_ctrl)
        history_rewards_std.append(eval_reward_std)
        print(
            f"Step {num_steps:>9d} | EvalReward {eval_reward:>9.2f} "
            f"| Run {eval_reward_run:>9.2f} | Ctrl {eval_reward_ctrl:>9.2f} "
            f"| Std {eval_reward_std:>8.2f} | SPS {sps:>7d}"
        )

    # Brax APIs vary across versions; map user args to whatever this version supports.
    train_sig = inspect.signature(sac_train.train)
    supported = set(train_sig.parameters.keys())

    def set_if_supported(name: str, value, kwargs: dict) -> None:
        if name in supported:
            kwargs[name] = value

    def set_first_supported(candidates: list[str], value, kwargs: dict) -> None:
        for name in candidates:
            if name in supported:
                kwargs[name] = value
                return

    train_kwargs = {}
    set_if_supported("environment", env, train_kwargs)
    set_if_supported("env", env, train_kwargs)
    set_if_supported("num_timesteps", args.timesteps, train_kwargs)
    set_if_supported("episode_length", args.episode_length, train_kwargs)
    set_if_supported("action_repeat", 1, train_kwargs)
    set_if_supported("num_envs", args.num_envs, train_kwargs)
    set_if_supported("num_eval_envs", args.num_eval_envs, train_kwargs)
    set_first_supported(["learning_rate", "lr"], args.learning_rate, train_kwargs)
    set_if_supported("batch_size", args.batch_size, train_kwargs)
    set_first_supported(["discounting", "discount"], args.discount, train_kwargs)
    set_first_supported(["entropy_cost", "init_alpha", "alpha_init"], args.entropy_cost, train_kwargs)
    set_if_supported("reward_scaling", args.reward_scaling, train_kwargs)
    set_if_supported("tau", args.tau, train_kwargs)
    set_if_supported("seed", args.seed, train_kwargs)
    set_if_supported("progress_fn", progress, train_kwargs)

    print(f"Env backend: {args.backend}")
    print(f"Using Brax SAC kwargs: {sorted(train_kwargs.keys())}")
    print("Starting Brax SAC train... first run may spend significant time in JIT compilation.")

    set_if_supported("num_evals", args.num_evals, train_kwargs)

    # Brax SAC is fully JAX-compiled and typically much faster than CPU gym loops.
    try:
        make_policy, params, _ = sac_train.train(**train_kwargs)
    except Exception as exc:
        msg = str(exc)
        if "cuSolver" in msg or "XlaRuntimeError" in msg:
            print("\nDetected GPU linear algebra backend failure (likely cuSolver under heavy parallelism).")
            print("Try one or more of these settings:")
            print("  1) Use a lighter pipeline: --backend spring")
            print("  2) Lower parallelism first: --num_envs 1024 --num_eval_envs 64")
            print("  3) Reduce batch size: --batch_size 1024")
            print("  4) Disable JAX memory preallocation before launch:")
            print("     export XLA_PYTHON_CLIENT_PREALLOCATE=false")
            print("     export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7")
            print("  5) If still failing, pin one GPU to verify stability:")
            print("     export CUDA_VISIBLE_DEVICES=0")
        raise

    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Save params as a JAX pytree checkpoint.
    try:
        import pickle

        with open("models/brax_sac_params.pkl", "wb") as f:
            pickle.dump(params, f)
        print("Saved params: models/brax_sac_params.pkl")
    except Exception as exc:
        print(f"Warning: failed to save params pickle: {exc}")

    if history_steps:
        import numpy as onp

        def smooth_1d(values: onp.ndarray, window: int) -> onp.ndarray:
            if window <= 1 or values.size < window:
                return values
            left = window // 2
            right = window - 1 - left
            padded = onp.pad(values, (left, right), mode="edge")
            kernel = onp.ones((window,), dtype=onp.float32) / window
            return onp.convolve(padded, kernel, mode="valid")

        rewards = onp.array(history_rewards, dtype=onp.float32)
        rewards_run = onp.array(history_rewards_run, dtype=onp.float32)
        rewards_ctrl = onp.array(history_rewards_ctrl, dtype=onp.float32)
        rewards_std = onp.array(history_rewards_std, dtype=onp.float32)

        smooth = smooth_1d(rewards, args.smooth_window)
        smooth_run = smooth_1d(rewards_run, args.smooth_window)
        smooth_ctrl = smooth_1d(rewards_ctrl, args.smooth_window)
        smooth_std = smooth_1d(rewards_std, args.smooth_window)

        best_idx = int(onp.argmax(rewards))
        best_step = history_steps[best_idx]
        best_reward = float(rewards[best_idx])

        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

        axes[0].plot(history_steps, rewards, color="tab:blue", alpha=0.35, label="Episode Reward")
        axes[0].plot(history_steps, smooth, color="tab:orange", label=f"Reward Smoothed({args.smooth_window})")
        axes[0].scatter([best_step], [best_reward], color="tab:red", s=35, label=f"Best {best_reward:.1f}")
        axes[0].set_title(f"Brax SAC Reward Curves ({args.env})")
        axes[0].set_ylabel("Total Reward")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(history_steps, rewards_run, color="tab:green", alpha=0.35, label="Reward Run")
        axes[1].plot(history_steps, smooth_run, color="tab:olive", label=f"Run Smoothed({args.smooth_window})")
        axes[1].plot(history_steps, rewards_ctrl, color="tab:purple", alpha=0.35, label="Reward Ctrl")
        axes[1].plot(history_steps, smooth_ctrl, color="tab:pink", label=f"Ctrl Smoothed({args.smooth_window})")
        axes[1].plot(history_steps, smooth_std, color="tab:brown", label=f"Reward Std Smoothed({args.smooth_window})")
        axes[1].set_xlabel("Environment Steps")
        axes[1].set_ylabel("Run/Ctrl/Std")
        axes[1].grid(alpha=0.3)
        axes[1].legend(ncol=2)

        fig.tight_layout()
        plt.savefig("plots/brax_sac_reward.png", dpi=200)
        plt.close(fig)
        print("Saved plot: plots/brax_sac_reward.png")

    # Keep a reference so linters do not strip these objects in some environments.
    _ = make_policy


if __name__ == "__main__":
    main()
