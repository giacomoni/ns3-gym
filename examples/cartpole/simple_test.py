#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym
import argparse
import ray
from ray import tune
from ns3gym import ns3env
import sys
import random
import numpy as np
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

def ns3gymapienv_creator(env_config):

    port = 5555 + env_config.worker_index
    simTime = 400 # seconds
    stepTime = 1  # seconds
    seed = 0 + env_config.worker_index
    simArgs = {"--simTime": simTime,
            "--testArg": 123}
    debug = False
    startSim = 1
    return ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)  # return an env instance

register_env("ns3-v0", ns3gymapienv_creator)

if __name__ == '__main__':

    env = sys.argv[1]
    num_workers = int(sys.argv[2])
    seed = int(sys.argv[3])
    
    random.seed(seed)
    np.random.seed(seed)

    ray.init(num_cpus=32)

    #env = gym.make()
    stepIdx = 0

    port = 5555
    simTime = 400 # seconds
    stepTime = 1  # seconds
    seed = 0
    simArgs = {"--simTime": simTime,
            "--testArg": 123}
    debug = False
    startSim = 1

    env_config = {"port":port, "stepTime":stepTime, "startSim":startSim, "simSeed":seed, "simArgs":simArgs, "debug":debug}

    algo = (
    DQNConfig()
    .rollouts(num_rollout_workers=num_workers)
    .resources(num_gpus=0)
    .environment(env) # "ns3-v0"
    .build()
)

    while True:
        result = algo.train()
        print(result['episode_reward_mean'])
        if result['episode_reward_mean'] >= 450:
            break


    # config = {"env": env,
    #         "num_workers" : num_workers,
    #         # "env_config": {"iniPath": os.getenv('HOME') + "/raynet/configs/cartpole/cartpole.ini"},
    #         "evaluation_config": {
    #                             "explore": False

    #     },
    #     "horizon": 500,
    #         "seed":seed}

    # ray.tune.run(
    #     "DQN",  
    #     name=f"{env}_{num_workers}_{seed}",
    #     config=config, 
    #     stop={"episode_reward_mean": 450.0},
    #     time_budget_s=2000, 
    #     checkpoint_at_end= True)
    

    ray.shutdown()