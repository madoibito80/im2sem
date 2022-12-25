# coding: utf-8

import sys
import os
import numpy as np
import pickle
from PIL import Image

sys.path.append(".")
import bullet_gym
import util
import collect_offline2

class Environment():
    def __init__(self, envtype, task, optp=False, lowp=False):
        self.envtype = envtype
        self.task = task
        self.optp = optp
        self.lowp = lowp
        if self.envtype == 'vizdoom' and self.task == 'shooting':
            self.dim_in = 2
            self.dim_out = 3
            self.img_size = 64
            sys.path.append('../env/vizdoom/')
            import doom
            self.env = doom.ShootWrapper(inspect=False)
            self.len_episode = self.env.MAX_STEP
            assert self.len_episode == 50
            self.n_ch = 1

        if self.envtype == 'pybullet' and self.task == 'kuka_grasp':
            self.dim_in = 9
            self.dim_out = 7
            sys.path.append("../env/pybullet/kuka_grasp")
            from kuka_diverse_object_gym_env import KukaDiverseObjectEnv
            self.img_size = 64
            self.len_episode = 42
            self.n_ch = 3
            self.env = KukaDiverseObjectEnv(
                isDiscrete=True,
                renders=False,
                height=self.img_size,
                width=self.img_size,
                maxSteps=self.len_episode,
                isTest=False,
            )
            #self.env = ObserveElapsedSteps(self.env, self.len_episode)
            # KukaDiverseObjectEnv internally asserts int actions
            #self.env = CastAction(self.env, int)
            env_seed = 0
            self.env.seed(int(env_seed))

        if self.envtype == 'pybullet' and 'halfcheetah' in self.task:
            self.len_episode = 1000
            self.n_ch = 3 # モーメントを除いた次元数
            self.img_size = 64
            self.dim_out = 6
            self.dim_in = 26

            if self.optp:
                # 最適方策
                self.env = collect_offline2.EnvBulletOptimal()
                self.env.env.venv.envs[0].env.mom = 0

            elif self.lowp:
                # 低性能方策
                self.env = collect_offline2.EnvBulletOptimal(model_path="../rl-baselines3-zoo/test02/ppo/HalfCheetahBulletEnv-v0_3/HalfCheetahBulletEnv-v0.zip")
                self.env.env.venv.envs[0].env.mom = 0
                #self.env.env.venv.envs[0].env.env._max_episode_steps = 250
            else:
                # ランダム方策
                self.env = collect_offline2.EnvBulletOptimal(fullrandom=True)
                self.env.env.venv.envs[0].env.mom = 0
                #self.env.env.venv.envs[0].env.env._max_episode_steps = 250
                



            """
            else:
                import gym
                import pybullet_envs
                if True:
                    new_env = bullet_gym.make_gym(env_id='HalfCheetahBulletEnv-v0')
                    new_env._max_episode_steps = 50
                    self.env = bullet_gym.Wrapper(env=new_env)
                    self.env.mom = 0
            
                #self.env = bullet_gym.load_env(seed=0, algo="ppo", env_id='HalfCheetahBulletEnv-v0')
            """

        
        if self.envtype == 'pybullet' and self.task == 'walker':
            raise NotImplementedError
            import gym
            import pybullet_envs
            self.env = gym.make(id='Walker2DBulletEnv-v0')
            self.len_episode = 1000
            self.n_ch = 3
            self.img_size = 64
            self.dim_out = 6
            self.dim_in = 22
            self.env.env._cam_dist = 1.3 # default=3



    def reset(self):
        if self.envtype == 'vizdoom':
            _ = self.env.init()
            state = self.env.state2img(self.env.game.get_state())
            state = util.resize(state, self.img_size)
            state = state.transpose((2,0,1))
            sem_state = self.env.game2sensor()[0]
            return (state, sem_state)
        if self.task == 'kuka_grasp':
            (state, sem_state) = self.env.reset()
            return (state, sem_state)
        """
        if self.task == 'halfcheetah' and not self.optp:
            sem_state = self.env.reset()
            state = self.env.get_current_img()
            return (state, sem_state)
        """
        if 'halfcheetah' in self.task:# and self.optp:
            self.env.reset()
            return (None, None)




    def step(self, action):
        if self.envtype == 'vizdoom':
            (state_next, action, state, is_end, reward, sem_state_next, sem_state) = self.env.random_observe(action)

            # transform
            state = util.resize(state, self.img_size)
            state_next = util.resize(state_next, self.img_size)
            state = state.transpose((2,0,1))
            state_next = state_next.transpose((2,0,1))

            return (state, state_next, action, is_end, reward, sem_state, sem_state_next)

        if self.task == 'kuka_grasp':
            obss = self.env.get_observation()
            int_action = int(np.where(np.array(action)==True)[0][0])
            obss_next, reward, is_end, _ = self.env.step(int_action)

            return (obss[0], obss_next[0], action, is_end, reward, obss[1], obss_next[1])

        if 'halfcheetah' in self.task:
            return self.env.step(action)
        """
        if self.task == 'halfcheetah' and not self.optp:

            state = self.env.get_mulch_obs()
            sem_state = self.env.robot.calc_state()
            sem_state_next, reward, is_end, _ = self.env.step(action)

            state_next = self.env.get_mulch_obs()
            return (state, state_next, action, float(is_end*1.), reward, sem_state, sem_state_next)
        """






class Buffer():
    def __init__(self, dump_dir, max_files=50):
        self.n_file = 0
        self.c = 0
        self.dump_dir = dump_dir
        self.q_state = None
        self.max_files = max_files
        self.steppfile = 1000

    def set_emp(self, img_size, dim_in, dim_out, n_ch):
        self.q_state = np.zeros((0,3*n_ch,img_size,img_size))
        self.q_state_next = np.zeros((0,3*n_ch,img_size,img_size))
        self.q_action = np.zeros((0,dim_out))
        self.q_is_end = np.zeros((0))
        self.q_reward = np.zeros((0))
        self.q_sem_state = np.zeros((0, dim_in))
        self.q_sem_state_next = np.zeros((0, dim_in))
        return None

    def push(self, cont):
        if self.n_file >= self.max_files:
            return False

        (p_state, p_state_next, p_action, p_is_end, p_reward, p_sem_state, p_sem_state_next) = cont

        if self.q_state is None:
            img_size = p_state.shape[2]
            dim_in = p_sem_state.shape[1]
            dim_out = p_action.shape[1]
            n_ch = int(p_state.shape[1]/3)
            self.set_emp(img_size, dim_in, dim_out, n_ch)

        e = min(self.c+p_state.shape[0], self.steppfile)
        l = min(e-self.c, p_state.shape[0])
        self.q_state = np.r_[self.q_state, p_state[:l]]
        self.q_state_next = np.r_[self.q_state_next, p_state_next[:l]]
        self.q_action = np.r_[self.q_action, p_action[:l]]
        self.q_is_end = np.r_[self.q_is_end, p_is_end[:l]]
        self.q_reward = np.r_[self.q_reward, p_reward[:l]]
        self.q_sem_state = np.r_[self.q_sem_state, p_sem_state[:l]]
        self.q_sem_state_next = np.r_[self.q_sem_state_next, p_sem_state_next[:l]]

        if e == self.steppfile or self.n_file == 0:
            cont = (self.q_state, None, self.q_action, self.q_is_end, self.q_reward, self.q_sem_state, self.q_sem_state_next)
            with open(self.dump_dir+"/"+str(self.n_file), "wb") as f:
                pickle.dump(cont, f)
            print('dump: ', self.n_file, self.q_state.shape)

        if e == self.steppfile:
            self.n_file += 1
            self.q_state = p_state[l:]
            self.q_state_next = p_state_next[l:]
            self.q_action = p_action[l:]
            self.q_is_end = p_is_end[l:]
            self.q_reward = p_reward[l:]
            self.q_sem_state = p_sem_state[l:]
            self.q_sem_state_next = p_sem_state_next[l:]
        
        self.c = (self.c + p_state.shape[0]) % self.steppfile

        return True





def play_episode(env, policy, img=False):

    p_state = np.zeros((env.len_episode,3*env.n_ch,env.img_size,env.img_size))
    p_state_next = np.zeros((env.len_episode,3*env.n_ch,env.img_size,env.img_size))
    p_action = np.zeros((env.len_episode,env.dim_out))
    p_is_end = np.zeros((env.len_episode))
    p_reward = np.zeros((env.len_episode))
    p_sem_state = np.zeros((env.len_episode, env.dim_in))
    p_sem_state_next = np.zeros((env.len_episode, env.dim_in))


    (state, sem_state) = env.reset()
    for n_sample in range(env.len_episode):

        if img:
            action = policy(state)
        else:
            action = policy(sem_state)
        (state, state_next, action, is_end, reward, sem_state, sem_state_next) = env.step(action)

        # put
        p_state[n_sample] = state
        p_state_next[n_sample] = state_next
        p_action[n_sample] = np.array(action)*1
        p_is_end[n_sample] = is_end
        p_reward[n_sample] = reward
        p_sem_state[n_sample] = sem_state
        p_sem_state_next[n_sample] = sem_state_next

        if is_end == 1:
            p_state = p_state[:n_sample+1]
            p_state_next = p_state_next[:n_sample+1]
            p_action = p_action[:n_sample+1]
            p_is_end = p_is_end[:n_sample+1]
            p_reward = p_reward[:n_sample+1]
            p_sem_state = p_sem_state[:n_sample+1]
            p_sem_state_next = p_sem_state_next[:n_sample+1]
            break

    print("episode-reward: ", float(np.sum(p_reward)) , " len-episode: ", len(p_state))
    return (p_state, p_state_next, p_action, p_is_end, p_reward, p_sem_state, p_sem_state_next)

