import os
import random

import numpy as np
import torch as th

from dm_control import suite 
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

DOMAIN_NAME = "cartpole"
TASK_NAME = "swingup"


def make_cartpole_env(seed: int | None = None, monitor_logdir: str | None = None):
    """
    Create DeepMind Control cartpole-swingup using dm_control.suite,
    then wrap it as a Gymnasium env via Shimmy, then flatten observations.
    """
    # 1) Raw dm_control env
    dm_env = suite.load(domain_name=DOMAIN_NAME, task_name=TASK_NAME)

    # 2) Wrap to Gymnasium-compatible env
    env = DmControlCompatibilityV0(dm_env, render_mode=None)

    # 3) Flatten dict observations to a single Box for MlpPolicy
    env = FlattenObservation(env)

    # 4) Seeding
    if seed is not None:
        # DmControlCompatibilityV0 implements Gymnasium's seed via reset()
        env.reset(seed=seed)
        env.action_space.seed(seed)

    # 5) Optional Monitor wrapper to log episode rewards
    if monitor_logdir is not None:
        os.makedirs(monitor_logdir, exist_ok=True)
        monitor_path = os.path.join(monitor_logdir, "monitor.csv")
        env = Monitor(env, filename=monitor_path)

    return env


def train_one_seed(
    train_seed: int,
    eval_seed: int,
    total_timesteps: int = 150_000,
    n_eval_episodes: int = 10,
):
    """Train PPO on cartpole-swingup for a single training seed."""


    set_random_seed(train_seed)
    random.seed(train_seed)
    np.random.seed(train_seed)
    th.manual_seed(train_seed)

    log_dir = os.path.join("runs_ppo", f"seed_{train_seed}")
    os.makedirs(log_dir, exist_ok=True)


    train_env = make_cartpole_env(seed=train_seed, monitor_logdir=log_dir)

    eval_env = make_cartpole_env(seed=eval_seed, monitor_logdir=None)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    # PPO hyperparameters
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=os.path.join("runs_ppo", "tensorboard"),
        seed=train_seed,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    
    model.save(os.path.join(log_dir, "final_model"))

    train_env.close()
    eval_env.close()


def main():
    total_timesteps = 150_000
    eval_seed = 10

    for train_seed in (0, 1, 2):
        train_one_seed(train_seed, eval_seed, total_timesteps=total_timesteps)


if __name__ == "__main__":
    main()
