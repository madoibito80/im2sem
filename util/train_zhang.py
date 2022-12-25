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
import time


sys.path.append(".")
import model
import util
import util_policy
import policies
import train_f



def train(task, trial, use_vae, buffer, ns):

    try:
        os.mkdir("../dump/"+task+"/"+trial+"/model/f_"+buffer+"_"+str(ns))
    except:
        None

    chainer.config.train = True

    data = util.Loader("../dump/"+task+"/offline")

    dim_out = data.q_sem_state.shape[1]

    f = open('../dump/'+task+'/'+trial+'/model/tran',"rb")
    tran = pickle.load(f)
    f.close()

    dic = model.Discriminator(dim_in=dim_out)
    dic = dic.to_gpu()
    optimizer2 = chainer.optimizers.Adam(alpha=0.001)
    optimizer2.setup(dic)
    #optimizer2.add_hook(chainer.optimizer.GradientClipping(1.0))
    #optimizer2.add_hook(chainer.optimizer.WeightDecay(0.0001))


    # LOAD Pair Data
    with open("../dump/"+task+'/'+trial+'/f/data_random_point', "rb") as f:
        [q_state, q_sem_state, selected] = pickle.load(f)

    aug_steps = len(selected[0][1])
    q_state = q_state[:ns*aug_steps]
    q_sem_state = q_sem_state[:ns*aug_steps]




    if use_vae:
        with open("../dump/"+task+"/"+trial+"/model/vae", "rb") as f:
            vae = pickle.load(f)
        phi = model.Phi3(dim_in=vae.z_dim, dim_out=dim_out, encoder=vae)	
    else:
        phi = model.Phi(dim_out=s_d, n_ch=n_ch)
    
    
    phi = phi.to_gpu()
    optimizer1 = chainer.optimizers.Adam(alpha=0.001)
    optimizer1.setup(phi)


    lcurve = {"loss":[]}

    if 'shooting' == task:
        max_iter = 40000 # 1000*10の200epoch
    elif 'kuka_grasp' == task:
        max_iter = 40000
    elif 'halfcheetah' in task:
        max_iter = 100000
    else:
        raise NotImplementedError



    

    c = 0
    while c < max_iter:
        c += 1

        if c % int(max_iter/5) == 0: # total 5(4)times reduce 
            optimizer1.alpha /= 3.
            optimizer2.alpha /= 3.

        # Loss_pair
        if ns > 0:
            batch_size=50

            inx = np.random.randint(0,len(q_state),batch_size)
            x = q_state[inx]
            t = q_sem_state[inx]

            y = phi.forward(x)
            loss_pair = phi.lossfunc(y, t)
        else:
            loss_pair = chainer.Variable(cp.zeros(1).astype(cp.float32))

        # Loss_dyn
        (state, state_tp1, _, _, action) = data.draw(batch_size=50)
        y = phi.forward(state)
        y_tp1 = phi.forward(state_tp1)

        t_tp1 = tran.forward(y, action)

        loss_dyn = F.mean_absolute_error(y_tp1, t_tp1)

        # Loss_adv
        label = chainer.Variable(cp.ones(y.shape[0]).astype(cp.int32).reshape((-1,1)))
        ### label 1 represents real-ness
        loss_adv1 = dic.lossfunc(dic.forward(y), label)
        loss_adv2 = dic.lossfunc(dic.forward(t_tp1), label)
        
        lam3 = (ns>0) * 1. 
        if False:
            # default: 200-10-10-?
            loss = 200.*loss_dyn + 10.*loss_adv1 + 10.*loss_adv2 + lam3*loss_pair # original, but does not work
            loss_adv = loss_adv1 + loss_adv2
        else:
            loss_adv = loss_adv1
            loss = 1.*loss_dyn + 2.*loss_adv + lam3*loss_pair



        phi.cleargrads()
        tran.cleargrads()
        dic.cleargrads()
        loss.backward()
        optimizer1.update()



        ########## Dic Update Step
        if c % 1 == 0:
            (state0, _, _, _, _) = data.draw(batch_size=50)
            (_, _, x1, _, _) = data.draw(batch_size=50)

            x0 = phi.forward(state0)

            ## MAKE TEST REAL
            if False:
                    x1 = cp.random.rand(100).reshape((50,2)).astype(cp.float32)
                    x1[:,0] = 0.5
                    x1 = chainer.Variable(x1)

            z = F.concat((x0,x1), axis=0)
            label = cp.ones(z.shape[0]).astype(cp.int32).reshape((-1,1))
            label[:x0.shape[0]] = 0
            label = chainer.Variable(label)
            # label 0 represents fake-ness
            loss_dic = dic.lossfunc(dic.forward(z), label)
            
            dic.cleargrads()
            phi.cleargrads()
            loss_dic.backward()
            optimizer2.update()

            if c % 10 == 0:
                print(c, float(loss_dyn.data), float(loss_adv.data), float(loss_pair.data))




        if c % 1000 == 0: # max-1に+1した状態が最終iterなので最後にも呼ばれる

            with open("../dump/"+task+"/"+trial+"/model/f_"+buffer+"_"+str(ns)+"/f", "wb") as f:
                pickle.dump(phi, f)
                pickle.dump(data.scale, f)
                pickle.dump(data.bias, f)
            with open("../dump/"+task+"/"+trial+"/model/f_"+buffer+"_"+str(ns)+"/lcurve", "wb") as f:
                pickle.dump(lcurve, f)

        



def plot_scatter(task, trial):

    chainer.config.train = False

    with open("../dump/"+task+"/"+trial+"/model/f_zhang_"+str(0)+"/f", "rb") as f:
        phi = pickle.load(f)

    data_0 = util.Loader("../dump/"+task+"/offline")


    #fig, ax = plt.subplots(10, 10, figsize=(25, 25), sharex=True, sharey=True)
    #nrow = 10
    #ncol = nrow

    for k in range(10): # for file randomness
                data_0.load()
                data_0.normalize()
                (state, _, state2, _, _) = data_0.draw(batch_size=100)

                state = phi.forward(state)
                print(state.shape)
                state = cuda.to_cpu(state.data)
                state2 = cuda.to_cpu(state2.data)
                plt.scatter(state[:,0], state[:,1], c="red", alpha=0.2)

                plt.scatter(state2[:,0], state2[:,1], c="blue", alpha=0.2)

    plt.savefig("./scatter.png")
    print("plot")




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--task', type=str, default='shooting')
    parser.add_argument('--trial', type=str, default='0')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()


    cp.random.seed(int(args.trial))
    np.random.seed(int(args.trial))

    gpu_device = int(args.gpu)
    cuda.get_device(gpu_device).use()




    if args.mode == 'train':

        if args.task == 'shooting':
            nss = [0,10,80]
        elif args.task == 'kuka_grasp':
            nss = [0,10,100]
        elif 'halfcheetah' in args.task:
            nss = [0]
        else:
            raise NotImplementedError

        for ns in nss:
            for buffer in ["zhang"]:
                train(task=args.task, trial=args.trial, use_vae=True, buffer=buffer, ns=ns)
                time.sleep(30)


    if args.mode == 'scatter':
        plot_scatter(args.task, args.trial)
