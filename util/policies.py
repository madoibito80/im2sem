# coding: utf-8

import numpy as np
from chainer import cuda
import chainer.functions as F


class collect_wrapper():
  def __init__(self, policy):
    self.policy = policy
  def __call__(self, state):
    return self.policy(state)
  def new_eps(self):
    return None
  


def hand_crafted_shooting(x):
  d = x[0] - x[1]
  thr=0.09
  if d > thr:
    inx = 2
  elif d < -thr:
    inx = 1
  else:
    inx = 0

  y = [False]*3
  y[inx] = True
  return y


def hand_crafted_kuka_grasp(z):
  
    z = z.reshape(9)
    z[1] += 0.019
    xm = z[0] - z[6]
    xp = z[6] - z[0]
    ym = z[1] - z[7]
    yp = z[7] - z[1]

    cand = np.array([xm, xp, ym, yp])
    inx = int(np.argmax(cand) + 1)

    v = [False]*7
    v[inx] = True
    return v


class behaviour_shooting():

  def __init__(self):
    self.actions = [[True, False, False], [False, True, False], [False, False, True]]
    self.new_eps()


  def new_eps(self):
    rd = np.random.rand()
    if rd < 0.5:
      self.lr = 1
    else:
      self.lr = 2

    self.nc = 0
    self.nt = [3,20][np.random.randint(0,2)]
    return None


  def __call__(self, x):
    if self.nc == 0:
      self.nc = self.nt
                
      if self.lr == 1:
        self.lr = 2
      else:
        self.lr = 1

    self.nc -= 1
    action = self.actions[self.lr]
    return action



class behaviour_kuka_grasp():

  def __init__(self):
    self.c = 0
    self.inx = None

  def new_eps(self):
    return None

  def __call__(self, x):

    if self.c % 10 == 0:
      self.inx = np.random.randint(0,7)
    self.c += 1
    v = [False]*7
    v[self.inx] = True
    return v


class behaviour_halfcheetah():
  def __init__(self):
    return None

  def new_eps(self):
    return None

  def __call__(self, x):
    return np.random.randint(0,2,6)*2-1




class behaviour_walker():
  def __init__(self):
    return None

  def new_eps(self):
    return None

  def __call__(self, x):
    return np.random.rand(6)


