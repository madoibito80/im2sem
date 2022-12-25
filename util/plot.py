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
import bullet_gym



def proc():

    fig, ax = plt.subplots(3, 10, figsize=(25, 8), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.rcParams["font.size"] = 18

    #plt.gcf().text(0.05,0.8,"ylabel")#,rotation=90)
    #plt.subplots_adjust(0.2,0.2,0.9,0.9)
    
    for tr in range(3):
        target = ["random_point", "random_batch", "al"][tr]
        with open("../dump/shooting/0/f/data_"+target, "rb") as f:
            cont = pickle.load(f)

        state = cont[1].data

        st = 0
        batchsize = len(cont[-1][0][1])  
        
        for k in range(10):
            
            x = cuda.to_cpu(state[st:st+batchsize])
            st += batchsize

            for c in range(k,10):
                ax[tr][c].set_aspect('equal', adjustable='box')
                if k == c:
                    ax[tr][c].scatter(x[:,0], x[:,1], c="red", alpha=0.8, label="selected")
                elif k < c:
                    ax[tr][c].scatter(x[:,0], x[:,1], c="blue", alpha=0.8, label="pre-selected")

                
                if tr == 0:
                    ax[tr][c].set_title(str(c)+"-th round") # \n(random for all method)
                ax[tr][c].tick_params(labelbottom=False,
               labelleft=False,
               labelright=False,
               labeltop=False,
               bottom=False,
               left=False)

               



    #plt.legend()

    #plt.axis("off")
    plt.savefig("./prog.png",transparent=True)


#proc()


def proc2():

    hfont = {'fontname':'Helvetica'} 

    import matplotlib
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True

    fig, ax = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0., hspace=0.)
    plt.rcParams["font.size"] = 20

    #plt.gcf().text(0.05,0.8,"ylabel")#,rotation=90)
    #plt.subplots_adjust(0.2,0.2,0.9,0.9)
    
    for tr in range(3):
        target = ["random_point", "random_batch", "al"][tr]
        with open("../dump/shooting/0/f/data_"+target, "rb") as f:
            cont = pickle.load(f)

        state = cont[1].data

        st = 0
        batchsize = len(cont[-1][0][1])  
        
        if True:
            
                c = 0

                for i in range(10):
                    x = cuda.to_cpu(state[i*batchsize:(i+1)*batchsize])

                    ax[tr].set_aspect('equal', adjustable='box')
                    color = "#3BAF75"
                    ax[tr].scatter(x[:,0], x[:,1], c=color, alpha=0.8, s=50)
               
                
                #if tr == 0:
                title = ["CRAR", "Ours w/o AL", "Ours"]
                ax[tr].set_title(title[tr]) # \n(random for all method)
                #ax[tr].set_xlim(0,1)
                #ax[tr].set_ylim(0,1)
                ax[tr].tick_params(labelbottom=False,
               labelleft=False,
               labelright=False,
               labeltop=False,
               bottom=False,
               left=False)

               


    plt.subplots_adjust(left=0, right=1, bottom=0.03, top=0.87)
    #plt.legend()

    #plt.axis("off")
    plt.savefig("./prog.pdf")#,transparent=True)


def perf():

    tasks = ['halfcheetah_low']
    buffers = ['test_b', 'test_o']

    for task in tasks:
        for buffer in buffers:
            dirn = '../dump/'+task+'/'+buffer+'/'
            rews = []
            for i in range(5):
                fname = dirn+str(i)
                with open(fname, "rb") as f:
                    (_, _, _, q_is_end, q_reward, _, _) = pickle.load(f)
                
                r = 0
                for j in range(1000):
                    r += q_reward[j]
                    if q_is_end[j] == 1:
                        rews.append(r)
                        r = 0

            rews = np.array(rews)
            print(fname, " : ", np.mean(rews), " pm ", np.std(rews))


#perf()
#exit()


def table(task):

    crits = ['md_b', 'rl']
    methods = ['zhang', 'random_point', 'random_batch', 'al']
    alias = ['zhang et al.', 'CRAR', 'Ours w/o AL', 'Ours']

    if task == 'shooting':
        nss = [0,10,50]
        trials = [0,1,2,3,4]
    elif task == 'kuka_grasp':
        nss = [0,10,100]
        trials = [0,1,2,3,4]
    elif 'halfcheetah' in task:
        nss = [0,10,50]
        trials = [0,1,2,3,4]
    else:
        raise NotImplementedError

    for ns in nss:
        print('\hline')
        print('\multicolumn{'+str(len(crits)+1)+'}{c}{$|\mathcal{I}|='+str(ns)+'$} \\\\')
        print('\hline')
        for i in range(len(methods)):
            method = methods[i]
            for crit in crits:
                with open('../dump/'+task+'/result_'+crit, "rb") as f:
                    results = pickle.load(f)
                #print(results)
                key = 'f_'+method+'_'+str(ns)
                if not key in results.keys():
                    continue

                if crit == crits[0]:
                    row = alias[i] + ' & '

                results[key] = np.array(results[key])
                if task == 'shooting' and 'md' in crit:
                    results[key] *= 100.

                #print(key, len(results[key]))

                m = np.mean(results[key])
                s = np.std(results[key])

                dig = 2
                m = round(m,dig)
                s = round(s,dig)

                m = str(m)
                s = str(s)

                for j in range(2-len(m.split('.')[1])):
                    m += '0'

                for j in range(2-len(s.split('.')[1])):
                    s += '0'

                r = "$"+m+"\pm"+s+"$ & "
                row += r
            row = row[:-2]
            row += "\\\\"
            print(row)


def table_noisy(task):

    crits = ['md_b', 'rl']
    total = {}
    for crit in crits:
        with open('../dump/'+task+'/result_noisy_'+crit, "rb") as f:
            results = pickle.load(f)

        for k,v in results.items():
            v = np.array(v)
            if task == 'shooting' and 'md' in crit:
                v *= 100.
            m = np.mean(v)
            s = np.std(v)

            dig = 2
            m = round(m,dig)
            s = round(s,dig)

            m = str(m)
            s = str(s)

            total[k+"_"+crit] = m+"\pm"+s


    for k,v in total.items():
        print(k,v)




    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='table')
    parser.add_argument('--task', type=str, default='shooting')
    parser.add_argument('--alpha_anno', type=float, default=0.)
    parser.add_argument('--alpha_tran', type=float, default=0.)
    args = parser.parse_args()

    if args.mode == 'table':
        table(task=args.task)
    if args.mode == 'table_noisy':
        table_noisy(task=args.task)


