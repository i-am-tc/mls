from starlette.requests import Request
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from ray import serve
import gymnasium as gym
import numpy as np
import math


# https://docs.ray.io/en/latest/rllib/rllib-env.html#configuring-environments 
def lula_genesis(env_config):
    lula_env = gym.make("LunarLander-v2")
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
register_env("lula_env", lula_genesis)


@serve.deployment
class LulaAlgo:
    def __init__(self, checkpoint_path) -> None:
        cfg = DQNConfig().environment(env="lula_env").rollouts(num_rollout_workers=1).framework("torch")
        self.algo = cfg.build(env="lula_env")
        self.algo.restore(checkpoint_path)

    async def __call__(self, request: Request):
        json_input = await request.json()
        obs = json_input["observation"]
        action = self.algo.compute_single_action(obs)
        return {"action": int(action)}

best_chkpt_path = "/home/tc/ray_results/DQN_2024-01-14_06-58-52/DQN_lula_env_b3bf7_00000_0_2024-01-14_06-58-52/checkpoint_000028"
lula_algo = LulaAlgo.bind(best_chkpt_path)