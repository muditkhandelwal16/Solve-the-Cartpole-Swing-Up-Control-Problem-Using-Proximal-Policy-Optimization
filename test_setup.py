from dm_control import suite
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DMCWrapper(gym.Env):
    def __init__(self, domain, task, seed=None):
        self.env = suite.load(domain_name=domain, task_name=task)

        self._seed = seed

        obs_spec = self.env.observation_spec()
        obs_size = sum(np.prod(v.shape) for v in obs_spec.values())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action spec
        act_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=act_spec.minimum, high=act_spec.maximum, dtype=np.float32
        )

    def seed(self, seed=None):
        # dm_control requires manual seeding
        if seed is not None:
            self._seed = seed
        np.random.seed(self._seed)

    def _flatten_obs(self, obs):
        return np.concatenate([v.ravel() for v in obs.values()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        time_step = self.env.reset()
        obs = self._flatten_obs(time_step.observation)
        return obs, {}

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._flatten_obs(time_step.observation)
        reward = float(time_step.reward)
        terminated = time_step.last()
        truncated = False
        return obs, reward, terminated, truncated, {}


if __name__ == "__main__":
    env = DMCWrapper("cartpole", "swingup", seed=0)
    obs, info = env.reset()
    print("Obs shape:", obs.shape)
    print("Action space:", env.action_space)

    for _ in range(20):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        if term:
            obs, info = env.reset()

    print("Custom dm_control Gym Wrapper Works on Windows ✅")
