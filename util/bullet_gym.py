import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed


sys.path.append("../rl-baselines3-zoo/.")
import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict


import gym
import pybullet
import pybullet_envs
from pybullet_envs import env_bases

sys.path.append(".")
import util


import chainer
from chainer import cuda
import cupy as cp
import pickle



def myrender(self, mode='human', close=False):
    #target = ['torso', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot'] half cheetah
    #target = [x for x in self.parts.keys() if not 'link' in x and 'floor' != x]
    target = ['torso','bfoot','ffoot']
    parts_xyz = [self.parts[x].pose().xyz() for x in target]

    ret_hole = corender(self, mode, close, parts_xyz[0], dist=1.1)
    #for i in range(100):
    #    self._p.changeVisualShape(objectUniqueId=0, linkIndex=i, rgbaColor=[0,0,0,0], physicsClientId=self.physicsClientId)
    
    for i in range(1,len(target)):
        xyz = parts_xyz[i]
        if i == 2:
            xyz[0] += 0.
        else:
            xyz[0] += 0.2
        img = corender(self, mode, close, xyz, dist=0.8) # 0.65
        #shape = list(img.shape)
        #shape[-1] = 1
        #img = np.repeat(np.mean(img,axis=2).reshape(shape),3,axis=2)
        ret_hole = np.concatenate((ret_hole,img), axis=2)

    #ret_left = corender(self, mode, close, xoff=-0.75, dist=1.2)
    #ret_right = corender(self, mode, close)
    #ret_hole = corender(self, mode, close, yoff=0.3)
    """
    st = int(ret_right.shape[0]/2)-20
    ed = st+int(ret_right.shape[0]/2)
    ret_hole[int(ret_hole.shape[0]/2):,:,:] = ret_right[st:ed,:,:]
    ret_hole[int(ret_hole.shape[0]/2),:,:] = 0
    """
    """
    is_ground = self.robot.calc_state()[-6:]
    for i in range(6):
        if is_ground[i] == 1:
            ret_hole[i*10:(i+1)*10,:10,1:] = 0
    """
    return ret_hole

def corender(self, mode='human', close=False, xyz=None, dist=2.1):
  
    if mode == "human":
      self.isRender = True
    if self.physicsClientId>=0:
      self.camera_adjust()

    if mode != "rgb_array":
      return np.array([])

    base_pos = [0, 0, 0]
    if (hasattr(self, 'robot')):
      if (hasattr(self.robot, 'body_real_xyz')):
        base_pos = self.robot.body_real_xyz.copy()


    if not xyz is None:
        base_pos = xyz

    self._cam_dist = dist




    if (self.physicsClientId>=0):
      view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
      proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(self._render_width) /
                                                     self._render_height,
                                                     nearVal=0.1,
                                                     farVal=100.0)
      (_, _, px, _, _) = self._p.getCameraImage(width=self._render_width,
                                              height=self._render_height,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

      self._p.configureDebugVisualizer(self._p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    else:
      px = np.array([[[255,255,255,255]]*self._render_width]*self._render_height, dtype=np.uint8)
    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array


env_bases.MJCFBaseBulletEnv.render = myrender


def load_model(seed, algo, env_id, env, model_path = "../rl-baselines3-zoo/test01/ppo/HalfCheetahBulletEnv-v0_33/HalfCheetahBulletEnv-v0.zip"):
        off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]
        kwargs = dict(seed=seed)
        if algo in off_policy_algos:
            # Dummy buffer size as we don't need memory to enjoy the trained agent
            kwargs.update(dict(buffer_size=1))

        # Check if we are running python 3.8+
        # we need to patch saved model under python 3.6/3.7 to load them
        newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

        custom_objects = {}
        if newer_python_version:
            custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            }

        #folder = "rl-trained-agents"
        #log_path = os.path.join(*["../../rl-baselines3-zoo", folder, algo, f"{env_id}_{1}"])
        #ext = "zip"
        #model_path = os.path.join(log_path, f"{env_id}.{ext}")
        #model_path = "../../rl-baselines3-zoo/rl-trained-agents/ppo/HalfCheetahBulletEnv-v0_1/HalfCheetahBulletEnv-v0.zip"
        #model_path = "../../rl-baselines3-zoo/test01/ppo/HalfCheetahBulletEnv-v0_33/HalfCheetahBulletEnv-v0.zip"
        print(model_path)
       


        model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
        return model


class wrapped_policy():
    def __init__(self, model, deterministic=True):
        self.model = model
        self.state = None
        self.deterministic = deterministic

    def __call__(self, obs):
        action, self.state = self.model.predict(obs, state=self.state, deterministic=self.deterministic)
        return action



def make_gym(env_id):

        new_env = gym.make(id=env_id)
        new_env._max_episode_steps = 1000
        #new_env.env._cam_dist = 1.9
        new_env.env._cam_pitch = 0
        new_env.env._render_width = 64
        new_env.env._render_height = new_env.env._render_width

        return new_env



def load_env(seed, algo, env_id, phi=None, scale=None, bias=None):
        folder = "rl-trained-agents"
        log_path = os.path.join(*["../rl-baselines3-zoo", folder, algo, f"{env_id}_{1}"])

        stats_path = os.path.join(log_path, env_id)
        hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=False) 

        env = create_test_env(
        env_id,
        n_envs=1,
        stats_path=stats_path,
        seed=seed,
        log_dir=None,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs={},
        )

        new_env = make_gym(env_id)
        new_env = Wrapper(new_env, phi=phi, scale=scale, bias=bias)
        env.venv.envs[0].env = new_env


        return env



class Wrapper(gym.Env):
    """Wraps the environment to allow a modular transformation.

    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.

    .. note::

        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.

    """
    def __init__(self, env, img_size=64, mom=1, phi=None, scale=None, bias=None, add_noise=False):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        try:
            self.reward_range = self.env.reward_range
        except:
            None
        self.metadata = self.env.metadata
        self.img_size = img_size

        self.bef_imgs = []
        self.mom = mom
        self.phi = phi
        self.scale = scale
        self.bias = bias

        self.on_min = None
        self.on_max = None
        self.add_noise = add_noise

        
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    @property
    def spec(self):
        return self.env.spec

    @classmethod
    def class_name(cls):
        return cls.__name__
    """
    def step2(self, action):

        state = self.get_mulch_obs()
        sem_state = self.robot.calc_state()
        sem_state_next, reward, is_end, _ = self.step(action)

        state_next = self.get_mulch_obs()
        return (state, state_next, action, float(is_end*1.), reward, sem_state, sem_state_next)
    """

    def step_wait(self):
        retv = self.env.step_wait()

        if self.add_noise: # calc scale and add noise
            state = retv[0]
            mi = np.min(state,axis=0)
            ma = np.max(state,axis=0)


            if self.on_min is None:
                self.on_min = mi
                self.on_max = ma
            else:
                self.on_min = mi*(self.on_min>mi) + self.on_min*(self.on_min<=mi)
                self.on_max = ma*(self.on_max<ma) + self.on_max*(self.on_max>=ma)

            scale = self.on_max - self.on_min + 0.01
            eps = np.random.randn(len(state.flatten())).reshape(state.shape)
            eps[:,-1] = 0. # 最終次元は時間なので無視
            scale = scale.reshape((1,len(scale)))
            n = 0.1
            retv = list(retv)
            retv[0] += eps*(n/3)*scale

        return retv

    def step_async(self, actions):
        retv = self.env.step_async(actions)
        # allways return None
        return retv


    def step(self, action):

        self.bef_imgs.append(self.get_current_img())
        if len(self.bef_imgs) > self.mom:
            self.bef_imgs = self.bef_imgs[-self.mom:]

        retv = list(self.env.step(action))
        if not self.phi is None:
            retv[0] = self.convert()

        """
        if False:
            c = self.convert()
            #inx = [9,11,13,15,17,19] # 角速度
            inx = [3,5] # x,z加速度
            ninx = [x for x in range(26) if not x in inx] # 角速度以外
            retv[0][inx] = c[inx]
            #retv[0] = 1.0*(retv[0] - c) + c
            #retv[0][:-6] = c[:-6]
            #retv[0][-6:] = 0.#np.random.randint(0,2,6)
        """
        return retv

            

    def convert(self):
        state = self.get_mulch_obs()
        state = state.reshape((1,-1,64,64))

        #with open("../dump/halfcheetah/policy/traj", "ab") as f:
        #    pickle.dump(state, f)


        x = cuda.to_gpu(state).astype(cp.float32) / 255.
        x = chainer.Variable(x)
        s = self.phi(x)
        s = cuda.to_cpu(s.data)
        s = s.flatten()

        s *= cuda.to_cpu(self.scale).flatten()
        s += cuda.to_cpu(self.bias).flatten()


        return s

    def reset(self, **kwargs):
        self.bef_imgs = []
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def get_current_img(self):
        state = self.env.render(mode='rgb_array')
        #state = util.resize(state, self.img_size)
        state = state.transpose((2,0,1))
        return state

    def get_mulch_obs(self):
        while len(self.bef_imgs) < self.mom:
            action = np.zeros(self.env.action_space.shape)
            self.step(action)

        obs = self.get_current_img()
        for i in range(self.mom):
            obs = np.r_[self.bef_imgs[-i-1], obs.copy()]
            #obs2 = np.concatenate((self.bef_imgs[-i-1], obs), axis=0)

        #for i in range(self.mom):
        #    dff = self.get_current_img()-self.bef_imgs[-i-1]
        #    dff *= (dff>=0.)
        #    obs = np.r_[dff, obs]
        
        return obs.copy()


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    #parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="tqc", type=str, required=False, choices=list(ALGOS.keys()))
    #parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    #parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    #parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    #parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    #parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    #parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    #parser.add_argument(
    #    "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    #)
    #parser.add_argument(
    #    "--load-checkpoint",
    #    type=int,
    #    help="Load checkpoint instead of last model if available, "
    #    "you must pass the number of timesteps corresponding to it",
    #)
    #parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    #parser.add_argument(
    #    "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    #)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    #parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    #parser.add_argument(
    #    "--gym-packages",
    #    type=str,
    #    nargs="+",
    #    default=[],
    #    help="Additional external Gym environemnt package modules to import (e.g. gym_minigrid)",
    #)
    #parser.add_argument(
    #    "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    #)
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    #for env_module in args.gym_packages:
    #    importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo



    args.n_envs = 1

    set_random_seed(args.seed)



    env = load_env(seed=args.seed, algo=algo, env_id=env_id)
    model = load_model(seed=args.seed, algo=algo, env_id=env_id, env=env)

    obs = env.reset()


 
    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0

    policy = wrapped_policy(model=model)
    while True:
            #print(env.norm_obs, env.obs_keys, env.obs_rms, isinstance(env.obs_rms, dict))
            #action, state = model.predict(obs, state=state, deterministic=True)
            action = policy(obs)
            obs, reward, done, infos = env.step(action)

            if not args.no_render:
                env.render("human")

            episode_reward += reward[0]
            ep_len += 1


            if done:
                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    print(f"Episode Reward: {episode_reward:.2f}")
                    print("Episode Length", ep_len)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    episode_reward = 0.0
                    ep_len = 0
                    state = None
                    break
    env.close()


if __name__ == "__main__":
    main()
