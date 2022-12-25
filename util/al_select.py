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
import time

sys.path.append(".")
import model
import util



def random_select(n_select, task, trial, batchmode=True):

    selected = []
    data = util.Loader("../dump/"+task+"/offline")

    for i in range(n_select):
        while True:
            nfile = int(np.random.choice(data.l, 1)[0])
            data.load(nfile=nfile)
            data.normalize()
            if batchmode:
                inx = int(np.random.choice(np.arange(len(data.eps_cand)), 1)[0])
                data.episode_draw(c=inx)
                inx = list(data.inx.copy())
            else:
                inx = list(np.random.choice(data.cand, 1))

            c = [nfile, inx]
            #print(c)
            if not c in selected:
                break
        selected.append(c)

    return selected

def al_select(n_select, task, trial):


    SPAN = 10
    GPU = False

    chainer.config.train = False
    chainer.config.enable_backprop = False

    stt = time.time()

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()




    f = open('../dump/'+task+'/'+trial+'/model/vae',"rb")
    vae = pickle.load(f)
    f.close()


    selected = []

    data = util.Loader("../dump/"+task+"/offline")

    # ランダムにエピソードを1つ選択
    eps_inx = np.random.choice(np.arange(len(data.eps_cand)), 1)[0]
    (state, _, _, _, _) = data.episode_draw(c=eps_inx, span=SPAN)
    vae.encode(state)
    pre_z = vae.mean
    if not GPU:
        pre_z.to_cpu()

    selected.append([int(data.nfile), data.inx.copy().tolist()])


    for _ in range(n_select-1):
        # FREE
        chainer.cuda.memory_pool.free_all_blocks()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        data.load(nfile=0)
        data.normalize()
        oeds = []
        while True:
            episode = data.episode_draw(span=SPAN)
            if episode is None:
                break

            (state, _, _, _, _) = episode
            vae.encode(state)
            z = vae.mean

            if not GPU:
                z.to_cpu()

            # == OED ==
            all_mind = 0.
            for t in range(z.shape[0]):
                dif = (z[t] - pre_z).data
                norm = np.linalg.norm(dif, axis=1)
                all_mind += float(np.min(norm))

            oeds.append([data.nfile, data.inx.copy().tolist(), all_mind])


        # == IED ==
        oeds = sorted(oeds, reverse=True, key=lambda x: x[2])

        #for oed in oeds:
        #    print(oed)

        topk = int(len(oeds)*0.10) # top k%
        oeds = oeds[:topk]

        ieds = []

        for oed in oeds:
            data.load(nfile=oed[0])
            data.normalize()
            inx = oed[1][::SPAN]
            #print(inx)
            (state, _, _, _, _) = data.draw(batch_size=0,inx=inx)
            vae.encode(state)
            z = vae.mean

            all_d = 0.
            for t in range(z.shape[0]):
                dif = (z[t] - z).data
                norm = np.linalg.norm(dif, axis=1)
                all_d += float(np.sum(norm))

            ieds.append([oed[0], oed[1], all_d])

        #print("====IED====")
        ieds = sorted(ieds, reverse=True, key=lambda x: x[2])

        #for k in range(5):
        #    print(ieds[k])

        # concat
        ied = ieds[0]
        data.load(nfile=ied[0])
        data.normalize()
        inx = ied[1][::SPAN]
        #print(inx)
        (state, _, _, _, _) = data.draw(batch_size=0,inx=inx)
        vae.encode(state)
        z = vae.mean
        if not GPU:
            z.to_cpu()
        pre_z = F.concat((pre_z, z), axis=0)
        selected.append([ied[0], ied[1]])

        #print(selected)
        print(time.time()-stt, len(selected))
        
        

    return selected



def vconcat(selected, task, trial):

    chainer.config.train = False

    data = util.Loader("../dump/"+task+"/offline")
    q_state = None
    q_sem_state = None


    for eps in selected:
        data.load(nfile=eps[0])
        data.normalize()
        (state, _, sem_state, _, _) = data.draw(batch_size=0,inx=eps[1])

        state = F.reshape(state, (1,-1,64,64))
        sem_state = F.reshape(sem_state, (1,-1))


        """
        augmented = sem_state[0:1]
        last_state = sem_state[0:1]
        if batchmode:
            for i in range(state.shape[0]-1):
                last_state = tran.forward(last_state, action[i:i+1])
                augmented = F.concat((augmented, last_state),axis=0)
        else:
        
            augmented = sem_state
        """
        if q_state is None:
            q_state = state
            q_sem_state = sem_state
        else:
            q_state = F.concat((q_state, state), axis=0)
            q_sem_state = F.concat((q_sem_state, sem_state), axis=0)

    return [q_state, q_sem_state, selected]




def sem_var(task):

    data = util.Loader('../dump/'+task+'/source')

    for nfile in range(10):
        data.load(nfile=nfile)
        data.normalize()

        q_sem_state = cuda.to_cpu(data.q_sem_state)

        if nfile == 0:
            sem_state = q_sem_state
        else:
            sem_state = cp.r_[sem_state, q_sem_state]

    print(sem_state.shape)
    v = cp.var(sem_state,axis=0)
    print(v)

    return v


def noise(alpha,sem_var,size=1):
    # 11/2 alpha is std ratio
    alpha *= alpha
    # bef 11/2 alpha is var ratio

    c = cp.sqrt(alpha*sem_var)
    x = cp.random.normal(size=len(sem_var)*size).reshape((size,len(sem_var)))
    x *= c
    return x



def show_pp(task, trial='0', alpha=0.05, anno=False, tran=False):


    for buffer in ["random_batch", "random_point"]:
        if buffer == "random_point" and not anno:
            continue

        plt.clf()
        plt.figure(figsize=(6,6))


        fname_load = "../dump/"+task+'/'+trial+'/f/data_'+buffer
        with open(fname_load, "rb") as f:
            [q_state, q_sem_state, selected] = pickle.load(f)

        q_sem_state = q_sem_state.data
        y1 = cuda.to_cpu(q_sem_state)

        fname_dump = fname_load
        if alpha != 0. and anno:
            fname_dump += '_anno:' + str(alpha)
        if alpha != 0 and tran:
            fname_dump += '_tran:' + str(alpha)

        with open(fname_dump, "rb") as f:
            [q_state, q_sem_state, selected] = pickle.load(f)

        q_sem_state = q_sem_state.data
        y2 = cuda.to_cpu(q_sem_state)

        ep_len = len(selected[0][1])
        epss = np.arange(3*ep_len)
        plt.xlim(-0.15,1.15)
        plt.ylim(-0.15,1.15)
        plt.scatter(y1[epss,0], y1[epss,1], alpha=0.3, color="red")
        plt.scatter(y2[epss,0], y2[epss,1], alpha=0.3, color="blue")

        """
        for i in epss:
            lines = [y[epss], y2[epss]]
            lc = LineCollection(lines)
            fig, ax = plt.subplots()
            ax.add_collection(lc)
        """
        plt.title(fname_dump)
        plt.savefig("./scatter_"+buffer+"_"+str(anno)+str(tran)+".png")
    return None


def add_noise(task, trial='0', alpha=0.05, anno=False, tran=False):



    # calc variance of semantic space
    v = sem_var(task)


    for buffer in ["random_batch", "random_point"]:
        if buffer == "random_point" and not anno:
            continue

        fname_load = "../dump/"+task+'/'+trial+'/f/data_'+buffer
        with open(fname_load, "rb") as f:
            [q_state, q_sem_state, selected] = pickle.load(f)

        #q_sem_state = q_sem_state.data

        #y = cuda.to_cpu(q_sem_state)

        # calc normal dist noise with objective alpha scale
        l = int(np.sum([len(x[1]) for x in selected]))
        x = noise(alpha=alpha, sem_var=v, size=l)


        ve = cp.var(x,axis=0)
        print("var of epsilon: ", ve)
        ra = ve/v
        print("post alpha: ", ra)
        print("post alpha: ", ra*ra)
        print('=========')

        if False:
            noise_vec = cp.zeros((l,len(v)))

            ep_st = 0
            for i in range(len(selected)): # for each episode
                ep_len = len(selected[i][1])

                cum_noise = 0.
                for j in range(ep_st,ep_st+ep_len):
                    if (anno and j == ep_st) or (tran and j != ep_st):
                        cum_noise += x[j]

                    noise_vec[j] += cum_noise

                ep_st += ep_len
        else:
            noise_vec = {}

            # noise_vec = {[nfile]_[st-inx]} = cp.array([noise-vector])
            ep_st = 0
            for i in range(len(selected)): # for each episode
                ep_len = len(selected[i][1])
                noise_vec[str(selected[i][0])+'_'+str(selected[i][1][0])] = cp.zeros((ep_len,len(v)))
                cum_noise = 0.
                for j in range(ep_len):
                    if (anno and j == 0) or (tran and j != 0):
                        cum_noise += x[ep_st+j]

                    noise_vec[str(selected[i][0])+'_'+str(selected[i][1][0])][j] += cum_noise
                ep_st += ep_len


            if not q_state is None:
                print('adding to pickle sem:', buffer)
                ep_st = 0
                q_sem_state = q_sem_state.data
                for i in range(len(selected)):
                    ep_len = len(selected[i][1])
                    q_sem_state[ep_st:ep_st+ep_len] += noise_vec[str(selected[i][0])+'_'+str(selected[i][1][0])]
                    #print(selected[i])
                    ep_st += ep_len

                q_sem_state = chainer.Variable(q_sem_state)


        fname_dump = fname_load
        if alpha != 0. and anno:
            fname_dump += '_anno:' + str(alpha)
        if alpha != 0 and tran:
            fname_dump += '_tran:' + str(alpha)

        if q_state is None: # halfcheetah
            q_sem_state = noise_vec

        cont = [q_state, q_sem_state, selected]

        with open(fname_dump, "wb") as f:
            pickle.dump(cont, f)
            print(fname_dump)









if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='select')
    parser.add_argument('--task', type=str, default='shooting')
    parser.add_argument('--trial', type=str, default='0')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--alpha_anno', type=float, default=0.)
    parser.add_argument('--alpha_tran', type=float, default=0.)
    args = parser.parse_args()
    
    dirn = '../dump/'+args.task+'/'+args.trial

    cp.random.seed(int(args.trial))
    np.random.seed(int(args.trial))

    gpu_device = int(args.gpu)
    cuda.get_device(gpu_device).use()


    if args.mode == 'add_noise':

        add_noise(args.task, args.trial, alpha=float(args.alpha_anno), anno=True)
        add_noise(args.task, args.trial, alpha=float(args.alpha_tran), tran=True)

    elif args.mode == 'show_pp':

        show_pp(args.task, args.trial, alpha=float(args.alpha_anno), anno=True)
        show_pp(args.task, args.trial, alpha=float(args.alpha_tran), tran=True)

    elif args.mode == 'select':
        
        try:
            os.mkdir(dirn+"/f")
        except:
            raise Exception
        

        if args.task == 'shooting':
            ns = [100]
        elif args.task == 'kuka_grasp':
            ns = [100]
        elif 'halfcheetah' in args.task:
            ns = [50]
        else:
            raise NotImplementedError
        

        selected = random_select(ns[0], args.task, args.trial, batchmode=False)
        cont = vconcat(selected, args.task, args.trial)
        with open(dirn+"/f/data_random_point", "wb") as f:
            pickle.dump(cont, f)
        print(selected)


        
        selected = al_select(ns[0], args.task, args.trial)
        if 'halfcheetah' in args.task:
            cont = [None,None,selected]
        else:
            cont = vconcat(selected, args.task, args.trial)
        with open(dirn+"/f/data_al", "wb") as f:
            pickle.dump(cont, f)
        print(selected)

        
        
        selected = random_select(ns[0], args.task, args.trial)
        if 'halfcheetah' in args.task:
            cont = [None,None,selected]
        else:
            cont = vconcat(selected, args.task, args.trial)
        with open(dirn+"/f/data_random_batch", "wb") as f:
            pickle.dump(cont, f)
        print(selected)

        




        
 
        print("done")


    if args.mode == 'proc':
        plot_ploc(dirn)


