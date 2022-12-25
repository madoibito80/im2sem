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
from PIL import Image
import cupy as cp


sys.path.append(".")
import model
import util




"""
def train_transition(task, trial):

    chainer.config.train = True

    data_0 = util.Loader("../dump/"+task+"/offline")
    #data_0 = util.Loader("../dump/"+task+"/source")
    data_1 = util.Loader("../dump/"+task+"/source")

    s_d = data_1.q_sem_state.shape[1]
    a_d = data_1.q_action.shape[1]
    n_ch = int(data_0.q_state.shape[1]/3)

    vae = model.VAE(z_dim=32, n_ch=n_ch)
    tran = model.Transition_Continuous(s_d=s_d, a_d=a_d)

    vae = vae.to_gpu()
    tran = tran.to_gpu()

    optimizer1 = chainer.optimizers.Adam(alpha=0.0003)
    optimizer1.setup(vae)
    
    optimizer2 = chainer.optimizers.Adam(alpha=0.0003)
    optimizer2.setup(tran)
    #optimizer2.add_hook(chainer.optimizer.WeightDecay(0.0001))



    lcurve = {"loss_vae":[], "loss_tran":[]}


    dirn = '../dump/'+task+'/'+trial

    max_epoch = 10
    bef_epoch = 0
    while True:

        if data_0.epoch < max_epoch:
            # train VAE from offline dataset
            state = data_0.draw(batch_size=50)[0]
            y = vae.decode(vae.encode(state))
            loss_vae = vae.lossfunc(state, y)

            vae.cleargrads()
            loss_vae.backward()
            optimizer1.update()

            lcurve["loss_vae"].append(float(loss_vae.data))

        if data_1.epoch < max_epoch:
            # train transition from source dataset
            (_, _, state, state_next, action) = data_1.draw(batch_size=50)
            y = tran.forward(state, action)
            loss_tran = tran.lossfunc(y, state_next)
            
            tran.cleargrads()
            loss_tran.backward()
            optimizer2.update()

            lcurve["loss_tran"].append(float(loss_tran.data))

        print(lcurve["loss_vae"][-1])

        if data_0.epoch >= max_epoch and data_1.epoch >= max_epoch:
             break

        if bef_epoch != data_0.epoch:
            bef_epoch = data_0.epoch

            with open(dirn+"/model/vae", "wb") as f:
                pickle.dump(vae, f)
            with open(dirn+"/model/tran", "wb") as f:
                pickle.dump(tran, f)
            with open(dirn+"/model/lcurve", "wb") as f:
                pickle.dump(lcurve, f)
"""



def train_transition(task, trial):

    chainer.config.train = True

    data_1 = util.Loader("../dump/"+task+"/source")
    
    s_d = data_1.q_sem_state.shape[1]
    a_d = data_1.q_action.shape[1]

    tran = model.Transition_Continuous(s_d=s_d, a_d=a_d)
    tran = tran.to_gpu()

    optimizer1 = chainer.optimizers.Adam(alpha=0.001)
    optimizer1.setup(tran)
    #optimizer1.add_hook(chainer.optimizer.WeightDecay(0.0001))
    

    lcurve = {"loss":[]}


    dirn = '../dump/'+task+'/'+trial

    if task == 'shooting':
        max_epoch = 50
        assert len(data_1.l) == 50
    elif task == 'kuka_grasp':
        max_epoch = 50
        assert len(data_1.l) == 50
    elif 'halfcheetah' in task:
        max_epoch = 50
    else:
        raise NotImplementedError

    bef_epoch = 0
    while True:

        if data_1.epoch < max_epoch:
            (_, _, state, state_next, action) = data_1.draw(batch_size=50)
            y = tran.forward(state, action)
            loss = tran.lossfunc(y, state_next)

            tran.cleargrads()
            loss.backward()
            optimizer1.update()

            lcurve["loss"].append(float(loss.data))



        if data_1.epoch >= max_epoch:
             break

        if bef_epoch != data_1.epoch:
            print(lcurve["loss"][-1])
            print("epoch: ", data_1.epoch)
            bef_epoch = data_1.epoch

            with open(dirn+"/model/tran", "wb") as f:
                pickle.dump(tran, f)
            with open(dirn+"/model/lcurve_tran", "wb") as f:
                pickle.dump(lcurve, f)





def train(task, trial):

    chainer.config.train = True

    data_1 = util.Loader("../dump/"+task+"/source") # for caluc minmax
    data_0 = util.Loader("../dump/"+task+"/offline")
    
    n_ch = int(data_0.q_state.shape[1]/3)

    vae = model.VAE(z_dim=n_ch*32, n_ch=n_ch)

    vae = vae.to_gpu()

    optimizer1 = chainer.optimizers.Adam(alpha=0.001)
    optimizer1.setup(vae)
    #optimizer1.add_hook(chainer.optimizer.WeightDecay(0.0001))
    

    lcurve = {"loss_vae":[]}


    dirn = '../dump/'+task+'/'+trial

    if task == 'shooting':
        max_epoch = 200
        assert len(data_0.l) == 10
    elif task == 'kuka_grasp':
        max_epoch = 200
        assert len(data_0.l) == 10
    elif 'halfcheetah' in task:
        max_epoch = 50
    else:
        raise NotImplementedError

    #max_epoch = int(50 / (len(data_0.l)/100))
    bef_epoch = 0
    while True:

        if data_0.epoch < max_epoch:
            # train VAE from offline dataset
            state = data_0.draw(batch_size=50)[0]
            y = vae.decode(vae.encode(state))
            loss_vae = vae.lossfunc(state, y)

            vae.cleargrads()
            loss_vae.backward()
            optimizer1.update()

            lcurve["loss_vae"].append(float(loss_vae.data))



        if data_0.epoch >= max_epoch:
             break

        if bef_epoch != data_0.epoch:
            print(lcurve["loss_vae"][-1])
            print("epoch: ", data_0.epoch)
            bef_epoch = data_0.epoch

            with open(dirn+"/model/vae", "wb") as f:
                pickle.dump(vae, f)
            with open(dirn+"/model/lcurve_vae", "wb") as f:
                pickle.dump(lcurve, f)






def plot_lcurve(dirn):

    print("plot")

    f = open(dirn+'/model/lcurve',"rb")
    lcurve = pickle.load(f)
    f.close()


    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(8,8))
    #fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()
    #ax3 = ax1.twinx()

    l1 = ax1.plot(lcurve["loss_vae"][100:],label="VAE", color="r")
    l2 = ax2.plot(lcurve["loss_tran"][100:],label="Transition")

    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()
    #handler3, label3 = ax3.get_legend_handles_labels()

    ax1.legend(handler1, label1)
    ax2.legend(handler2, label2)
    #ax3.legend(handler3, label3)


    plt.savefig("./lcurve.png")


def plot_latent_scatter(task, trial):

    dirn = '../dump/'+task+'/'+trial
    f = open(dirn+'/model/vae',"rb")
    vae = pickle.load(f)
    f.close()


    data_0 = util.Loader("../dump/"+task+"/source")


    fig, ax = plt.subplots(10, 10, figsize=(25, 25), sharex=True, sharey=True)
    nrow = 10
    ncol = nrow

    for k in range(10): # for file randomness
        data_0.load()
        data_0.normalize()
        (state, _, _, _, _) = data_0.draw(batch_size=10)
        state = vae.encode(state)
        state = cuda.to_cpu(state.data)
        for i in range(nrow):
            for j in range(ncol):
                print(i,j,k)

                ax[i][j].scatter(state[:,i], state[:,j], c="red", alpha=0.2)

    plt.savefig("./latent_scatter.png")





def plot_rec_image(task, trial):


    dirn = '../dump/'+task+'/'+trial

    f = open(dirn+'/model/vae',"rb")
    vae = pickle.load(f)
    f.close()


    data_0 = util.Loader("../dump/"+task+"/offline")


    (state, _, _, _, _) = data_0.draw(batch_size=100)

    n_ch = int(state.shape[1]/3)


    w = 64

    dst = Image.new('RGB', ((w+2)*10*n_ch, (w+2)*10*2),(0,0,0))


    z = vae.encode(state)
    xs = vae.decode(z)



    for y in range(10):
        for p in range(10):
            for ch in range(n_ch):
                x = cuda.to_cpu(xs[y*10+p,ch*3:(ch+1)*3].data)
                x = (x*255).astype(np.uint8).transpose((1,2,0))
                x = Image.fromarray(x)
                xpo = n_ch*p*(w+2) + ch*(w+2)
                ypo = 2*y*(w+2)
                dst.paste(x, (xpo, ypo))

                x = cuda.to_cpu(state[y*10+p,ch*3:(ch+1)*3].data)
                x = (x*255).astype(np.uint8).transpose((1,2,0))
                x = Image.fromarray(x)
                xpo = n_ch*p*(w+2) + ch*(w+2)
                ypo = 2*y*(w+2)+(w+2)
                dst.paste(x, (xpo, ypo))




    dst.save("./rec.png")



def check_tran_error(task, trial):


    chainer.config.train = False


    f = open('../dump/'+task+'/'+str(trial)+'/model/tran',"rb")
    tran = pickle.load(f)
    f.close()


    data_1 = util.Loader("../dump/"+task+"/source")

    total_sample = 10000
    batch_size = 100
    score_1 = 0
    score_2 = 0
    score_3 = 0

    for it in range(int(total_sample/batch_size)):
        
        data_1.load()
        data_1.normalize()
        (_, _, state, g1, action) = data_1.draw(batch_size=batch_size)
        (_, _, g2, _, _) = data_1.draw(batch_size=batch_size)

        y = tran.forward(state, action)

        loss = F.sqrt(F.sum(F.square(y - g1), axis=1))
        score_1 += float(F.sum(loss).data)

        loss = F.sqrt(F.sum(F.square(g2 - g1), axis=1))
        score_2 += float(F.sum(loss).data)

        loss = F.sqrt(F.sum(F.square(state - g1), axis=1))
        score_3 += float(F.sum(loss).data)


    #score_1 /= total_sample
    print("predicted: ",score_1)


    #score_2 /= total_sample
    print("random: ",score_2)


    #score_2 /= total_sample
    print("equal: ",score_3)





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--task', type=str, default='shooting')
    parser.add_argument('--trial', type=str, default='0')
    #parser.add_argument('--nfile', type=str, default='0')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    dirn = '../dump/'+args.task+'/'+args.trial

    cp.random.seed(int(args.trial))
    np.random.seed(int(args.trial))

    gpu_device = int(args.gpu)
    cuda.get_device(gpu_device).use()


    if args.mode == 'tran':
        train_transition(args.task, args.trial)



    if args.mode == 'train':	
        try:
            os.mkdir(dirn+'/model')
        except:
            print("except")

        train(args.task, args.trial)



    if args.mode == 'lcurve':
        plot_lcurve(dirn)

    if args.mode == 'latent_scatter':
        plot_latent_scatter(args.task, args.trial)

    if args.mode == 'rec_image':
        plot_rec_image(args.task, args.trial)

    if args.mode == 'tran_error':
        check_tran_error(args.task, args.trial)