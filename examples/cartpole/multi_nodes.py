#import the simulation model with cart-pole
import gymnasium as gym
from gymnasium import spaces, logger
import numpy as np
import math
from ray.tune.registry import register_env
import ray
from ray import tune
import random
import sys
import os
import math
from ray.rllib.algorithms.dqn import DQNConfig
from ns3gym import ns3env
import time

def ns3gymapienv_creator(env_config):

    port = 5555 + env_config.worker_index
    simTime = 500 # seconds
    stepTime = 1  # seconds
    seed = 0 + env_config.worker_index
    simArgs = {"--simTime": simTime,
            "--testArg": 123}
    debug = False
    startSim = 1
    return ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)  # return an env instance

register_env("ns3-v0", ns3gymapienv_creator)

if __name__ == '__main__':

    env = "ns3-v0"
    nodes = 2
    seed = 10


    random.seed(seed)
    np.random.seed(seed)

    ray.init(address='auto')
    

    algo = (
    DQNConfig()
    .rollouts(num_rollout_workers=nodes*8-1)
    .resources(num_gpus=0)
    .environment(env) # "ns3-v0"
    .build()
)

    t_start = time.time()
    now = time.time()
    while True:
        print(f"Total elpsed: {(now - t_start)}")
        result = algo.train()
        print(result['num_env_steps_sampled'])
        if result['num_env_steps_sampled'] >= 500000:
            break
        now = time.time()
    ray.shutdown()

