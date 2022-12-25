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



"""
def main(dirn, buffer, task, gt):
    

    if task == 'shooting':
        p = policies.hand_crafted_shooting
        env = util_policy.Environment(envtype='vizdoom',task='shooting')
    elif task == 'kuka_grasp':
        p = policies.hand_crafted_kuka_grasp
        env = util_policy.Environment(envtype='pybullet',task='kuka_grasp')
    else:
        raise NotImplementedError


    chainer.config.train = True

    with open(dirn+'/f/data_'+buffer, "rb") as f:
        [q_state, q_sem_state, q_augmented, eds] = pickle.load(f)


    dim_out = q_sem_state.shape[1]

    
    span = [10,50,100]#,150,1000,5000] # num of annotation
    # al: 100 / 5000point
    # rand-batch: 1-150 / 7500point
    # rand-point: 1-5000 / 5000point
    span = [x for x in span if x <= len(eds)]
    print(span)

    
    for i in span:

        ed = eds[i-1] # use 0:ed
        print(ed)
        x = chainer.Variable(q_state[:ed].data)
        if gt:
            t = q_sem_state
        else:
            t = q_augmented
        t = t[:ed]

        phi = model.Phi(dim_out=dim_out, task=task)
        phi = phi.to_gpu()
        optimizer1 = chainer.optimizers.Adam(alpha=0.001)
        optimizer1.setup(phi)
        optimizer1.add_hook(chainer.optimizer.WeightDecay(0.0001))
        
        for it in range(60000):
            if x.shape[0] >= 100:
                inx = np.random.randint(0,x.shape[0],100)
                x2 = x[inx]
                t2 = t[inx]
            else:
                x2 = x
                t2 = t
            y = phi.forward(x2)
            loss = phi.lossfunc(y, t2)
            phi.cleargrads()
            loss.backward()
            optimizer1.update()
            if it % 100 == 0:
                loss = float(loss.data)
                print(it,loss)

        md = eval_md(phi)
        print("md", md)


        policy = lambda x: composite_policy(x, phi, p)
        r = eval_policy(policy, env)
        print("r", r)


        with open(dirn+"/f/results/"+buffer+"_"+str(i)+"_"+str(gt), "wb") as f:
            pickle.dump([buffer, i, ed, md, r], f)
            pickle.dump(phi, f)


    phi = phi.to_gpu()
"""



def train(task, trial, use_vae, buffer, ns, aug_steps=-1):


    chainer.config.train = True
    data = util.Loader("../dump/"+task+"/offline")

    try:
        os.mkdir("../dump/"+task+"/"+trial+"/model/f_"+buffer+"_"+str(ns))
    except:
        None

    use_loader = False

    if buffer == 'star':
        use_loader = True
    else:
        with open("../dump/"+task+'/'+trial+'/f/data_'+buffer, "rb") as f:
            [q_state, q_sem_state, selected] = pickle.load(f)


        len_eps = len(selected[0][1])

        if q_state is None: # halfcheetah batch
            l = [x[0] for x in selected]
            l = l[:ns]
            data.l = l
            use_loader = True

            data.load()
            data.normalize()
            # CAUTION: this impl depends 1file = 1eps
        else:
            q_state = q_state[:ns*len_eps]
            q_sem_state = q_sem_state[:ns*len_eps]


    s_d = data.q_sem_state.shape[1]
    n_ch = int(data.q_state.shape[1]/3)

    if use_vae:
        with open("../dump/"+task+"/"+trial+"/model/vae", "rb") as f:
            vae = pickle.load(f)
        phi = model.Phi3(dim_in=vae.z_dim, dim_out=s_d, encoder=vae)	
    else:
        phi = model.Phi(dim_out=s_d, n_ch=n_ch)
    
    
    phi = phi.to_gpu()
    optimizer1 = chainer.optimizers.Adam(alpha=0.0005)
    optimizer1.setup(phi)
    
    lcurve = {"loss":[]}

    if 'shooting' in dirn:
        max_iter = 40000 # 1000*10の200epoch
    elif 'kuka_grasp' in dirn:
        max_iter = 40000
    elif 'halfcheetah' in dirn:
        max_iter = 100000
    else:
        raise NotImplementedError


    # augmentation step modifi
    text_as = ""
    if aug_steps != -1:
        text_as = "_AS("+str(aug_steps)+")"
        os.mkdir("../dump/"+task+"/"+trial+"/model/f_"+buffer+"_"+str(ns)+text_as)
        print('dir made')


    c = 0
    while c < max_iter:
        c += 1

        batch_size=50

        if use_loader:

            if aug_steps != -1:
                inx = np.random.choice(data.cand[:aug_steps], size=aug_steps)
            else:
                inx = None

            (x, _, t, _, _) = data.draw(batch_size=batch_size, inx=inx)

            if ':' in buffer: # noisy halfcheetah
                noise = q_sem_state[str(data.nfile)+'_0'][data.inx]
                t += noise

        else:
            inx = np.random.randint(0,len(q_state),batch_size)
            x = q_state[inx]
            t = q_sem_state[inx]

        y = phi.forward(x)
        loss = phi.lossfunc(y, t)

        phi.cleargrads()
        loss.backward()
        optimizer1.update()

        lcurve["loss"].append(float(loss.data))
        #print(lcurve["loss"][-1])

        if c % 1000 == 0:

            print(buffer, c)

            with open("../dump/"+task+"/"+trial+"/model/f_"+buffer+"_"+str(ns)+text_as+"/f", "wb") as f:
                pickle.dump(phi, f)
                pickle.dump(data.scale, f)
                pickle.dump(data.bias, f)
            with open("../dump/"+task+"/"+trial+"/model/f_"+buffer+"_"+str(ns)+text_as+"/lcurve", "wb") as f:
                pickle.dump(lcurve, f)







def plot(dirn):
    al_types = ["al", "random_point", "random_batch"]
    trials = [1,2,3]
    pairs = [10,50,100]
    gts = [True, False]

    target = "md"

    plt.figure(figsize=(10,5))
    for al_type in al_types:
        for gt in gts:
            y = {"md":{}, "r":{}}
            for pair in pairs:
                y["md"][pair] = []
                y["r"][pair] = []
                for trial in [1,2,3]:
                    fname = dirn+str(trial)+"/f/results/"+al_type+"_"+str(pair)+"_"+str(gt)
                    with open(fname, "rb") as f:
                        [buffer, i, ed, md, r] = pickle.load(f)

                    y["md"][pair].append(md)
                    y["r"][pair].append(r)
            
            ys = np.zeros((len(pairs), 2))
            c = 0
            for pair in pairs:
                ys[c][0] = np.mean(y[target][pair])
                ys[c][1] = np.std(y[target][pair])
                c += 1

            plt.plot(pairs, ys[:,0], label=al_type+"_"+str(gt), alpha=0.9)
            plt.fill_between(pairs, y1=ys[:,0]-ys[:,1], y2=ys[:,0]+ys[:,1], alpha=0.2)
            print(al_type, gt, y[target])
    plt.subplots_adjust(left=0.1, right=0.6, bottom=0.1, top=0.96)
    print("plot")
    plt.title(target)
    #plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
    #plt.xlim(-1,1050)
    #plt.ylim(-1,2)
    plt.savefig("./res.png")

"""
def test(dirn):

    
    data = util.Loader(dirn+"/buffer/offline")
    data.load()
"""


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    #parser.add_argument('--env', type=str, default='vizdoom')
    parser.add_argument('--task', type=str, default='shooting')
    parser.add_argument('--trial', type=str, default='0')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--alpha_anno', type=float, default=0.)
    parser.add_argument('--alpha_tran', type=float, default=0.)
    parser.add_argument('--aug_steps', type=int, default=-1)
    parser.add_argument('--star', type=int, default=0)
    args = parser.parse_args()

    dirn = '../dump/'+args.task+'/'+args.trial


    cp.random.seed(int(args.trial))
    np.random.seed(int(args.trial))

    gpu_device = int(args.gpu)
    cuda.get_device(gpu_device).use()


    if args.task == 'shooting':
            #nss = [50,80]
            nss = [10,50]
    elif args.task == 'kuka_grasp':
            nss = [10,100]
    elif 'halfcheetah' in args.task:
            nss = [10,50]
    else:
            raise NotImplementedError

    if args.mode == 'noisy-train':

        alpha_anno = float(args.alpha_anno)
        alpha_tran = float(args.alpha_tran)
        ns = nss[1]
        
        buffers = ["random_batch_tran:"+str(alpha_tran)]

        if alpha_anno != 0.:
            buffers.extend(["random_batch_anno:"+str(alpha_anno), "random_point_anno:"+str(alpha_anno)])


        for buffer in buffers:
            train(task=args.task, trial=args.trial, use_vae=True, buffer=buffer, ns=ns, aug_steps=int(args.aug_steps))
            time.sleep(30)



    elif args.mode == 'train':

        if args.star == 0: # for train star only
            for ns in nss:
                for buffer in ["random_point","random_batch","al"]:
                    train(task=args.task, trial=args.trial, use_vae=True, buffer=buffer, ns=ns)
                    time.sleep(30)
        else:
            print("train f star")
            train(task=args.task, trial=args.trial, use_vae=True, buffer='star', ns=100000)


    if args.mode == 'plot':
        plot('../dump/'+args.task+'/')
