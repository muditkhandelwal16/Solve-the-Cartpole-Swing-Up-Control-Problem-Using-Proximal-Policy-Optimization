import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import load_results

LOG_ROOT = "runs_ppo"


def _load_training_rewards(seed: int) -> np.ndarray:
    """
    Load per-episode rewards from Monitor logs for a given seed.
    Assumes PPO training wrote monitor.csv into runs_ppo/seed_X/.
    """
    log_dir = os.path.join(LOG_ROOT, f"seed_{seed}")
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found for seed {seed}: {log_dir}")

    # load_results reads all *monitor.csv files in the folder and returns a DataFrame
    df = load_results(log_dir)
    if "r" not in df.columns:
        raise RuntimeError(f"No reward column 'r' found in monitor logs for seed {seed}")

    rewards = df["r"].values
    return rewards


def plot_training_curve(seeds=(0, 1, 2), window: int = 10) -> None:
    """
    Plot training episode return vs episode index (mean ± std across seeds).
    Optionally apply a moving-average smoothing over episodes.
    """
    rewards_per_seed = [np.asarray(_load_training_rewards(s)) for s in seeds]
    min_len = min(len(r) for r in rewards_per_seed)
    rewards_per_seed = [r[:min_len] for r in rewards_per_seed]

    # Shape: (n_seeds, n_episodes)
    rewards_matrix = np.stack(rewards_per_seed, axis=0)

    if window > 1:
        def smooth(x):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[window:] - cumsum[:-window]) / window

        rewards_matrix = np.stack([smooth(r) for r in rewards_matrix], axis=0)
        episodes = np.arange(rewards_matrix.shape[1])
    else:
        episodes = np.arange(min_len)

    mean_rewards = rewards_matrix.mean(axis=0)
    std_rewards = rewards_matrix.std(axis=0)

    fig, ax = plt.subplots()
    ax.plot(episodes, mean_rewards, label="mean return")
    ax.plot(episodes, mean_rewards + std_rewards, linestyle="--", label="mean ± std")
    ax.plot(episodes, mean_rewards - std_rewards, linestyle="--")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("PPO – DMC Cartpole Swingup (training)")
    ax.legend()
    fig.tight_layout()

    out_path = "ppo_cartpole_training_curve.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved training curve to {out_path}")


def plot_eval_curve(seeds=(0, 1, 2)) -> None:
    """
    Plot evaluation return vs environment steps (mean ± std across training seeds).

    Uses the evaluations.npz files created by EvalCallback in each seed directory.
    """
    all_means = []
    timesteps = None

    for seed in seeds:
        log_dir = os.path.join(LOG_ROOT, f"seed_{seed}")
        eval_path = os.path.join(log_dir, "evaluations.npz")

        if not os.path.isfile(eval_path):
            raise FileNotFoundError(
                f"evaluations.npz not found for seed {seed}: {eval_path}\n"
                f"Did you run ppo_cartpole_train.py long enough to trigger EvalCallback?"
            )

        data = np.load(eval_path, allow_pickle=True)
        ts = np.asarray(data["timesteps"], dtype=np.int64)

        if timesteps is None:
            timesteps = ts
        else:
            if not np.array_equal(timesteps, ts):
                raise RuntimeError("Evaluation timesteps do not match across seeds")

        # data["results"] is an array of shape (n_evals, n_eval_episodes)
        results = data["results"]
        mean_per_eval = np.array([np.mean(ep_rews) for ep_rews in results])
        all_means.append(mean_per_eval)

    all_means = np.stack(all_means, axis=0)
    mean_rewards = all_means.mean(axis=0)
    std_rewards = all_means.std(axis=0)

    fig, ax = plt.subplots()
    ax.plot(timesteps, mean_rewards, label="mean eval return")
    ax.plot(timesteps, mean_rewards + std_rewards, linestyle="--", label="mean ± std")
    ax.plot(timesteps, mean_rewards - std_rewards, linestyle="--")

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Return")
    ax.set_title("PPO – DMC Cartpole Swingup (evaluation)")
    ax.legend()
    fig.tight_layout()

    out_path = "ppo_cartpole_eval_curve.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved evaluation curve to {out_path}")


if __name__ == "__main__":
    plot_training_curve()
    plot_eval_curve()
