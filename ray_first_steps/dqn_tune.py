import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQNConfig
import gymnasium as gym
import numpy as np
import math

# https://docs.ray.io/en/latest/rllib/rllib-env.html#configuring-environments 
def lula_genesis(env_config):
    lula_env = gym.make("LunarLander-v2", render_mode="human", turbulence_power=1.5, enable_wind=False, wind_power=5.0)
    # Had to expand x & y, pygame was giving obs beyond oring range from source
    low = np.array(
    [
        -2.0,
        -2.0,
        -5.0,
        -5.0,
        -math.pi,
        -10.0,
        -0.0,
        -0.0,
    ]
    ).astype(np.float32)
    high = np.array(
        [
            2.0,
            2.0,
            5.0,
            5.0,
            math.pi,
            10.0,
            1.0,
            1.0,
        ]
    ).astype(np.float32)
    lula_env.observation_space = gym.spaces.Box(low, high)
    lula_env.reset(seed=1)
    return lula_env

###

if __name__ == "__main__":
    register_env("lula_env", lula_genesis)
    ray.init(num_gpus=1, num_cpus=10)
    cfg = DQNConfig().environment(env="lula_env").rollouts(num_rollout_workers=8)
    tune.run("DQN", config=cfg, checkpoint_freq=10)