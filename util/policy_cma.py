# coding: utf-8

import cma
import numpy as np
from chainer import cuda
import chainer.functions as F

class CMAES:
  '''CMA-ES wrapper.'''
  def __init__(self, dim_in,      # dim of input for policy
               dim_out,
               sigma_init=0.5,       # initial standard deviation
               popsize=16,           # population size
               weight_decay=0.01, # weight decay coefficient
               mean_init=None):    

    self.z_dim = dim_in
    self.a_dim = dim_out
    self.num_params = (self.z_dim + 1) * self.a_dim
    self.sigma_init = sigma_init
    self.popsize = popsize
    self.weight_decay = weight_decay
    self.solutions = None
    self.reward_table_result = []

    if mean_init is None:
      mean_init = self.num_params * [0]

    self.es = cma.CMAEvolutionStrategy( mean_init,
                                        self.sigma_init,
                                        {'popsize': self.popsize,
                                        })

  def hand_crafted(self, z):
    z = z.reshape(9)
    xm = z[0] - z[6]
    xp = z[6] - z[0]
    ym = z[1] - z[7]
    yp = z[7] - z[1]

    cand = np.array([xm, xp, ym, yp])
    inx = int(np.argmax(cand) + 1)

    #if z[inx] < 0.01:
    #  print(z[inx])
    #  inx = 0

    v = [False]*self.a_dim
    v[inx] = True
    return v


  def policy(self, z, i):

    if True:
      return self.hand_crafted(z)

    if i == -1:
      w = self.current_param()
    else:
      w = self.solutions[i]
    b = w[-self.a_dim:]
    w = w[:-self.a_dim].reshape((self.a_dim, self.z_dim))
    z = z.reshape((1,self.z_dim))

    logit = F.linear(w,z).reshape(self.a_dim) + b
    logit = F.reshape(logit, (1,self.a_dim))

    h = int(F.argmax(logit).data)
    v = [False]*logit.shape[1]
    v[h] = True
    return v

  def rms_stdev(self):
    sigma = self.es.result[6]
    return np.mean(np.sqrt(sigma*sigma))

  def compute_weight_decay(self, weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

  def ask(self, sigma_fac=1.):
    '''returns a list of parameters'''
    self.solutions = np.array(self.es.ask(sigma_fac=sigma_fac))
    return self.solutions
  """
  def update(self):
    self.tell(self.reward_table_result)
    self.reward_table_result = []
  """
  def tell(self, reward_table_result):
    reward_table = -np.array(reward_table_result)
    if self.weight_decay > 0:
      l2_decay = self.compute_weight_decay(self.weight_decay, self.solutions)
      reward_table += l2_decay
    self.es.tell(self.solutions, (reward_table).tolist()) # convert minimizer to maximizer.

  def current_param(self):
    return self.es.result[5] # mean solution, presumably better with noise

  def set_mu(self, mu):
    pass

  def best_param(self):
    return self.es.result[0] # best evaluated solution

  def result(self): # return best params so far, along with historically best reward, curr reward, sigma
    r = self.es.result
    return (r[0], -r[1], -r[1], r[6])

