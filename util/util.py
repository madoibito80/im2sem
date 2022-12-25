# coding: utf-8

import pickle
import os
import re
import numpy as np
import cupy as cp
import chainer
from chainer import cuda
from PIL import Image




class Loader():
    def __init__(self, dirname):
        self.dirname = dirname
        l = os.listdir(dirname)
        self.l = [x for x in l if re.match('[0-9]+',x)]
        self.c = 0
        self.epoch = 0
        self.nf = 0
        self.nfile = None
        self.inx = None

        if 'kuka_grasp' in self.dirname:
            self.aug_steps = 40
            self.mom = 0
        elif 'shooting' in self.dirname:
            self.aug_steps = 50
            self.mom = 0
        elif 'halfcheetah' in self.dirname:
            self.aug_steps = 1000
            self.mom = 1
        else:
            raise NotImplementedError


        r = self.min_max()
        self.bias = r[0]
        self.scale = r[1]-r[0]+0.01
        print(self.bias, self.scale)

        self.load()
        self.normalize()

    def min_max(self):
        
        dirname = self.dirname.replace("offline","source").replace("test_b","source").replace("test_o","source")
        if os.path.exists(dirname+'/minmax'):
            print('exists pre-calculated mini max: ',dirname+'/minmax')
            with open(dirname+'/minmax', 'rb') as f:
                (to_mi, to_ma) = pickle.load(f)
            return to_mi, to_ma
        
        for nfile in range(len(self.l)):
            print(nfile)
            #with open(fname, "rb") as f:
            #    (_, _, _, _, _, q_sem_state, _) = pickle.load(f)
            self.load(nfile=nfile)
            q_sem_state = self.q_sem_state

            mi = np.min(q_sem_state, axis=0)
            ma = np.max(q_sem_state, axis=0)

            if nfile == 0:
                to_mi = mi
                to_ma = ma
            else:
                to_mi = mi*(to_mi>mi) + to_mi*(to_mi<=mi)
                to_ma = ma*(to_ma<ma) + to_ma*(to_ma>=ma)

        to_mi = cuda.to_gpu(to_mi).reshape((1,-1)).astype(cp.float32)
        to_ma = cuda.to_gpu(to_ma).reshape((1,-1)).astype(cp.float32)

        with open(self.dirname+'/minmax', 'wb') as f:
            pickle.dump((to_mi, to_ma), f)

        return to_mi, to_ma


    def load(self, nfile=None):
        #print('load start...')
        self.nf += 1 # 読み込んだファイルの累計数
        self.epoch = int((self.nf-1)/len(self.l)) # 完了したエポック数
        self.c = 0 # drawカウンタ

        if nfile is None:
            nfile = self.l[np.random.randint(0,len(self.l))]

        self.nfile = nfile
        fname = self.dirname + '/' + str(nfile)
        with open(fname, "rb") as f:
            (self.q_state, self.q_state_next, self.q_action, self.q_is_end, self.q_reward, self.q_sem_state, self.q_sem_state_next) = pickle.load(f)
        
        self.cand = np.where(self.q_is_end==0)[0]




        # エピソードの開始インデックスの集合
        # ファイルの開始はエピソードの開始であることを前提
        self.eps_cand = [x for x in range(1000) if x%self.aug_steps==0]
        self.eps_cand = [x for x in self.eps_cand if np.sum(self.q_is_end[x:x+self.aug_steps-1])==0]

    
        
        # Bufferで節約したので，(1)momの複製
        if self.mom >= 2:
            raise NotImplementedError

        if self.mom == 1:
            q_state = self.q_state.copy()
            q_state_mom = np.concatenate((q_state[0:1], q_state[:-1]), axis=0) # ファイル区切り=エピソード区切りであることを仮定
            self.q_state = np.concatenate((q_state_mom, self.q_state),axis=1)

        # Bufferで節約したので，(2)stp1の複製
        self.q_state_next = np.zeros(self.q_state.shape)

        #print('load end !')
        return None


    def normalize(self):
        self.q_state = cuda.to_gpu(self.q_state).astype(cp.float32) / 255.
        self.q_state_next = cuda.to_gpu(self.q_state_next).astype(cp.float32) / 255.
        self.q_sem_state = (cuda.to_gpu(self.q_sem_state).astype(cp.float32) - self.bias) / self.scale
        self.q_sem_state_next = (cuda.to_gpu(self.q_sem_state_next).astype(cp.float32) - self.bias) / self.scale
        self.q_action = cuda.to_gpu(self.q_action).astype(cp.float32)


    def draw(self, batch_size, inx=None):
        if self.c >= self.q_state.shape[0]:
            self.load()
            self.normalize()
        self.c += batch_size

        if inx is None:
            inx = self.cand[np.random.randint(0,len(self.cand),batch_size)]
        
        self.inx = inx

        cont = [self.q_state[inx], self.q_state_next[inx], self.q_sem_state[inx], self.q_sem_state_next[inx], self.q_action[inx]]
        cont = [chainer.Variable(x) for x in cont]
        return cont

    def episode_draw(self, c=None, span=1):

        if not self.c < len(self.eps_cand):
            if len(self.l) > self.nfile+1:
                self.load(nfile=self.nfile+1) # episodeの巡回は非ランダム
                self.normalize()
            else:
                return None

        if c is None: # 現在loadされているファイルのエピソードカウンタの指すエピソードを選択
            c = self.c
            self.c += 1


        st = self.eps_cand[c]
        ed = st+self.aug_steps

        inx = np.arange(st,ed)

        
        self.inx = inx

        ## with span
        inx = inx[::span]

        cont = [self.q_state[inx], self.q_state_next[inx], self.q_sem_state[inx], self.q_sem_state_next[inx], self.q_action[inx]]
        cont = [chainer.Variable(x) for x in cont]
        return cont

        


def resize(np_img, img_size):
  img = Image.fromarray(np_img.astype(np.uint8))
  img = img.resize((img_size, img_size))
  np_img = np.array(img)
  return np_img
        




if __name__ == '__main__':
    # test
    dirname = '../dump/shooting/0/buffer/source'
    dataset = Loader(dirname)
    for i in range(1):
        dataset.draw(batch_size=50)
