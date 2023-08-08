#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym
from gymnasium import spaces, logger
from omnetbind import OmnetGymApi
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

class OmnetGymApiEnv(gym.Env):
    def __init__(self,env_config):
        
        self.action_space = spaces.Discrete(2)
        self.runner = OmnetGymApi()
        self.env_config = env_config

        
        high = np.array(
            [
                2.4 * 2,
                np.finfo(np.float32).max,
                (12 * 2 * math.pi / 360) * 2,
                np.finfo(np.float32).max,],
            dtype=np.float64,)
        self.observation_space = spaces.Box(-high, high, dtype=np.float64)

       
    def reset(self):

        original_ini_file = self.env_config["iniPath"]

        with open(original_ini_file, 'r') as fin:
            ini_string = fin.read()
        
        ini_string = ini_string.replace("HOME",  os.getenv('HOME'))

        with open(original_ini_file + f".worker{os.getpid()}", 'w') as fout:
            fout.write(ini_string)

        self.runner.initialise(original_ini_file + f".worker{os.getpid()}")
        obs = self.runner.reset()

        obs = np.asarray(list(obs['cartpole']),dtype=np.float32)
        return  obs

    def step(self, action):
        actions = {'cartpole': action}
        theta_threshold_radians = 12 * 2 * math.pi / 360
        x_threshold = 2.4
        obs, rewards, dones, info_ = self.runner.step(actions)
        reward = round(rewards['cartpole'],4)
        obs = obs['cartpole']

        if (obs[0] < x_threshold * -1) or (obs[0] > x_threshold) or (obs[2] < theta_threshold_radians * -1) or (obs[2] > theta_threshold_radians):
            dones['cartpole'] = True
            reward = 0

        if dones['cartpole']:
             self.runner.shutdown()
             self.runner.cleanup()
       
        obs = np.asarray(list(obs),dtype=np.float32)
    
        return  obs, reward, dones['cartpole'], {}

def omnetgymapienv_creator(env_config):
    return OmnetGymApiEnv(env_config)  # return an env instance

register_env("OmnetGymApiEnv", omnetgymapienv_creator)

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

    env = sys.argv[1]
    num_workers = int(sys.argv[2])
    seed = int(sys.argv[3])
    
    random.seed(seed)
    np.random.seed(seed)

    ray.init(num_gpus=0)

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
