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


# shooting r 0.46
# kuka grasp r 0.68


def composite_policy(state, phi, policy, scale, bias):
    state = state.reshape((1,-1,64,64))
    x = cuda.to_gpu(state).astype(cp.float32) / 255.
    x = chainer.Variable(x)
    s = phi(x)
    s = cuda.to_cpu(s.data)
    s = s.flatten()

    s *= cuda.to_cpu(scale).flatten()
    s += cuda.to_cpu(bias).flatten()

    action = policy(s)
    return action

def eval_policy(policy, env):

    n_repeat = 50
    r = 0.
    for i in range(n_repeat):
        print(i)
        cont = util_policy.play_episode(env=env, policy=policy, img=True)
        r += float(np.sum(cont[4])) / n_repeat
    return r


def eval_policy2(policy, env):

    n_repeat = 50
    rs = []
    for i in range(n_repeat):
        obs = env.reset() # 開始時のみsemanticsを取得する
        r = 0.
        for t in range(1010):
            action = policy(obs)
            obs, reward, done, infos = env.step(action)
            r += float(reward)
            if done:
                print(r)
                rs.append(r)
                break
    print("mean: ", np.mean(rs))
    print("std: ", np.std(rs))

    env.close()

    return float(np.mean(rs))



def check_md(path, task, mode):

    chainer.config.train = False

    with open(path, "rb") as f:
        phi = pickle.load(f)
        scale = pickle.load(f)
        bias = pickle.load(f)

    target = mode.replace("md_","test_")
    # expect "md_o" or "md_b"

    data = util.Loader("../dump/"+task+"/"+target)
    md = eval_md(phi, data)

    return md




def eval_md(phi, data):

    md_all = 0.
    c = 0
    for i in range(len(data.l)):
            data.load(nfile=i)
            data.normalize()

            inx = data.cand
            (x, _, t, _, _) = data.draw(batch_size=0, inx=inx)
            y = phi.forward(x)

            l = F.square(y - t) # こっちのtはscale, bias適用済み
            l = F.sum(l, axis=1)
            l = F.sum(l)
            l = float(l.data)
            md_all += l
            c += len(inx)

    md_all /= c
    print(md_all)
    return md_all



def check_ad(path, task, mode):

    chainer.config.train = False

    with open(path, "rb") as f:
        phi = pickle.load(f)
        scale = pickle.load(f)
        bias = pickle.load(f)

    target = mode.replace("ad_","test_")
    # expect "ad_o" or "ad_b"

    if task == 'shooting':
        policy = policies.hand_crafted_shooting
    elif task == 'kuka_grasp':
        policy = policies.hand_crafted_kuka_grasp
    else:
        raise NotImplementedError

    scale = cuda.to_cpu(scale).flatten()
    bias = cuda.to_cpu(bias).flatten()

    data = util.Loader("../dump/"+task+"/"+target)
    ad = eval_ad(phi, data, policy, scale, bias)

    return ad


def eval_ad(phi, data, policy, scale, bias):

    md_all = 0.
    c = 0
    for i in range(len(data.l)):
            data.load(nfile=i)
            data.normalize()

            inx = data.cand
            (x, _, ts, _, _) = data.draw(batch_size=0, inx=inx)

            for i in range(len(inx)):
                y = phi.forward(x[i:i+1])
                y = y.data.flatten()
                t = ts[i:i+1].data.copy().flatten()

                y = cuda.to_cpu(y)
                t = cuda.to_cpu(t)

                y *= scale
                y += bias
                t *= scale
                t += bias

                y = policy(y)
                t = policy(t)

                y = np.array(y)*1.
                t = np.array(t)*1.

                l = F.square(y - t) # こっちのtはscale, bias適用済み
                l = F.sum(l)
                l = float(l.data)
                md_all += l
                c += 1

    md_all /= c
    print(md_all)
    return md_all



def check_rl(path, task):

    chainer.config.train = False

    with open(path, "rb") as f:
        phi = pickle.load(f)
        scale = pickle.load(f)
        bias = pickle.load(f)


    if task == 'shooting':
        p = policies.hand_crafted_shooting
        env = util_policy.Environment(envtype='vizdoom',task='shooting')
    elif task == 'kuka_grasp':
        p = policies.hand_crafted_kuka_grasp
        env = util_policy.Environment(envtype='pybullet',task='kuka_grasp')
    elif 'halfcheetah' in task:
        algo = 'ppo'
        env = bullet_gym.load_env(seed=0, algo=algo, env_id='HalfCheetahBulletEnv-v0', phi=phi, scale=scale, bias=bias)
        model = bullet_gym.load_model(seed=0, algo=algo, env_id='HalfCheetahBulletEnv-v0', env=env)
        p = bullet_gym.wrapped_policy(model)
    else:
        raise NotImplementedError



    if task == 'shooting' or task == 'kuka_grasp':
        policy = lambda x: composite_policy(x, phi, p, scale, bias)
        r = eval_policy(policy, env)
    elif 'halfcheetah' in task:
        policy = p
        r = eval_policy2(policy, env)
    else:
        raise NotImplementedError

    return r




def main(path, task):
    
    chainer.config.train = False

    with open(path, "rb") as f:
        phi = pickle.load(f)
        scale = pickle.load(f)
        bias = pickle.load(f)


    """
    md = eval_md(phi, dirn)
    print("====MD====")
    for i in range(len(md)):
        print(i, md[i])
    """







def plot(dirn):
    al_types = ["random_point", "random_batch", "al"]
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

"""
def specific():
    
    path = '../dump/'+task+'/'+trial+'/model/'+target+'/f'

    gpu_device = int(args.gpu)
    cuda.get_device(gpu_device).use()

    if args.mode == 'check':
        main(dirn, path, task=args.task)

    if args.mode == 'plot':
        plot('../dump/'+args.task+'/')
"""





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='calc')
    parser.add_argument('--task', type=str, default='shooting')
    #parser.add_argument('--crit', type=str, default='rl')
    parser.add_argument('--wait', type=str, default='0')
    parser.add_argument('--noisy', type=bool, default=False)
    parser.add_argument('--aug_steps', type=int, default=-1)
    args = parser.parse_args()

    time.sleep(int(args.wait))

    crits = ['md_b', 'rl']#, 'ad_b', 'ad_o'] 'md_o'

    trials = [0,1,2,3,4]

    for crit in crits:
        results = {}
        for trial in trials:
        #for trial in [0,1,2]:

            gpu = int(trial%2)
            gpu_device = int(gpu)
            cuda.get_device(gpu_device).use()

            base = '../dump/'+args.task+'/'+str(trial)+'/model'

            l = os.listdir(base)
            l = [x for x in l if 'f_' in x]

            if args.noisy:
                alphas = {"anno":[0.04,0.15,0.3], "tran":[0.01,0.04,0.1]}
                l = [x for x in l if ':' in x]


                l2 = []
                for key,als in alphas.items():
                    print(key, als)
                    for alpha in als:
                        for x in l:
                            if key+":"+str(alpha) in x:
                                l2.append(x)
                l = l2




            if args.aug_steps != -1:
                text_as = "_AS(" #+str(args.aug_steps)+")"
                l = [x for x in l if text_as in x]



            for target in l:
                
                path = base+'/'+target+'/f'
                print(path)

                if crit == 'rl':
                    res = check_rl(path=path, task=args.task)
                elif 'md_' in crit:
                    res = check_md(path=path, task=args.task, mode=crit)
                elif 'ad_' in crit:
                    res = check_ad(path=path, task=args.task, mode=crit)
                else:
                    raise NotImplementedError

                if trial == trials[0]:
                    results[str(target)] = [res]
                else:
                    results[str(target)].append(res)


                time.sleep(10)


        print(results)
        if args.noisy:
            with open('../dump/'+args.task+'/result_noisy_'+crit, "wb") as f:
                pickle.dump(results, f)

        else:
            with open('../dump/'+args.task+'/result_'+crit, "wb") as f:
                pickle.dump(results, f)

    



