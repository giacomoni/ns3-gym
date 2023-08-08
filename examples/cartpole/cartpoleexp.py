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
register_env("OmnetGymApiEnv", omnetgymapienv_creator)

if __name__ == '__main__':

    env = sys.argv[1]
    num_workers = int(sys.argv[2])
    seed = int(sys.argv[3])


    random.seed(seed)
    np.random.seed(seed)

    ray.init(num_cpus=64)
    
    env_config = {"iniPath": os.getenv('HOME') + "/raynet/configs/cartpole/cartpole.ini"}

    algo = (
    DQNConfig()
    .rollouts(num_rollout_workers=num_workers)
    .resources(num_gpus=0)
    .environment(env, env_config=env_config) # "ns3-v0"
    .build()
)

    while True:
        result = algo.train()
        print(result['episode_reward_mean'])
        if result['episode_reward_mean'] >= 450:
            break

    ray.shutdown()


