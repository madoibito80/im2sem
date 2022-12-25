# coding: utf-8

import sys
import argparse
import chainer
import chainer.functions as F
import pickle
import os
import matplotlib.pyplot as plt
from chainer import cuda
import numpy as np
import cupy as cp
from PIL import Image
import re


sys.path.append(".")
import model
import util
import util_policy
import policies
import bullet_gym



algo = 'ppo'

class EnvBulletOptimal():

    def __init__(self, model_path=None, fullrandom=False, addnoise=False):
        self.fullrandom = fullrandom
        self.addnoise = addnoise
        self.env = bullet_gym.load_env(seed=0, algo=algo, env_id='HalfCheetahBulletEnv-v0')

        self.c = 0

        if model_path is None:
            self.model = bullet_gym.load_model(seed=0, algo=algo, env_id='HalfCheetahBulletEnv-v0', env=self.env)
        else:
            self.model = bullet_gym.load_model(seed=0, algo=algo, env_id='HalfCheetahBulletEnv-v0', env=self.env, model_path=model_path)

        self.p = bullet_gym.wrapped_policy(self.model)
        self.reset()

    def reset(self):
        self.policy_state = self.env.reset()

    def __call__(self, *args): # behaviour policy
        if self.fullrandom:
            self.c += 1
            if self.c % 100 == 0:
                self.new_eps()
            action = [self.action.copy() * np.random.rand(6)]
        else:
            action = self.p(self.policy_state)
            if self.addnoise:
                noise = np.random.randn(6) / 3
                action += noise
                action = np.clip(action, -1, 1)
        return action

    def new_eps(self): # behaviour policy
        print('action flip')
        self.c = 0
        self.action = np.random.randint(0,2,6)*2-1
        return None


    def step(self, action):
        state = self.env.venv.envs[0].env.get_mulch_obs()
        sem_state = self.env.venv.envs[0].env.robot.calc_state()

        self.policy_state, reward, is_end, _ = self.env.step(action)

        state_next = self.env.venv.envs[0].env.get_mulch_obs()
        sem_state_next = self.env.venv.envs[0].env.robot.calc_state()

        return (state, state_next, action, float(is_end*1.), reward, sem_state, sem_state_next)






def main(dirn, task):
    
    chainer.config.train = False


    if task == 'shooting':
        raise NotImplementedError
    elif task == 'kuka_grasp':
        raise NotImplementedError
    elif task == 'halfcheetah':
        env = bullet_gym.load_env(seed=0, algo=algo, env_id='HalfCheetahBulletEnv-v0')
        model = bullet_gym.load_model(seed=0, algo=algo, env_id='HalfCheetahBulletEnv-v0', env=env)
        p = bullet_gym.wrapped_policy(model)
    else:
        raise NotImplementedError



    
    if task == 'shooting' or task == 'kuka_grasp':
        raise NotImplementedError
    elif task == 'halfcheetah':
        policy = p
        r = eval_policy2(policy, env)
    else:
        raise NotImplementedError

    print("r", r)







if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--buffer', type=str, default='source')
    parser.add_argument('--task', type=str, default='halfcheetah')
    args = parser.parse_args()

    dirn = '../dump/'+args.task


    main(dirn, task=args.task)

