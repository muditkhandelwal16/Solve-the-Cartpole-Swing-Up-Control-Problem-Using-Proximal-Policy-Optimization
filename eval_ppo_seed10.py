import os
import numpy as np

from dm_control import suite
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

DOMAIN_NAME = "cartpole"
TASK_NAME = "swingup"

LOG_ROOT = "runs_ppo"


def make_cartpole_env(seed: int = 10):
    """
    Create the DMC cartpole-swingup env exactly like in training:
    dm_control.suite -> DmControlCompatibilityV0 -> FlattenObservation.
    """
    # 1) Load dm_control env
    dm_env = suite.load(domain_name=DOMAIN_NAME, task_name=TASK_NAME)

    # 2) Wrap to Gymnasium-compatible env
    env = DmControlCompatibilityV0(dm_env, render_mode=None)

    # 3) Flatten dict observation into a single Box
    env = FlattenObservation(env)

    # 4) Seed for deterministic evaluation
    env.reset(seed=seed)
    env.action_space.seed(seed)

    return env


def evaluate_best_model(train_seed: int, eval_seed: int = 10, n_episodes: int = 20):
    log_dir = os.path.join(LOG_ROOT, f"seed_{train_seed}")
    model_path = os.path.join(log_dir, "best_model.zip")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = make_cartpole_env(seed=eval_seed)
    model = PPO.load(model_path)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_episodes,
        deterministic=True,
    )

    print(
        f"PPO best_model (train seed {train_seed}) on env seed {eval_seed}: "
        f"{mean_reward:.2f} ± {std_reward:.2f} over {n_episodes} episodes"
    )

    env.close()


if __name__ == "__main__":
    evaluate_best_model(train_seed=0, eval_seed=10, n_episodes=20)
